"""Shiftr core scheduling engine.

Pipeline (each step is an ordinary, testable function):
    1. load_staff / load_availability   who can work, when, at what wage
    2. load_bookings                    confirmed future covers (the MINIMUM demand)
    3. load_walkin_history              Oct 2025 walk-ins by weekday (the EXTRA demand)
    4. build_demand                     people in the building per half-hour slot
    5. build_requirements               staff needed per station per slot, two tiers:
                                          req_min  = bookings only  -> must always be met
                                          req_full = bookings + forecast walk-ins
    6. build_schedule                   PuLP integer programme: cheapest rota that meets
                                        req_min, covers req_full where possible, and obeys
                                        availability and Working Time Regulations rules
    7. coverage_report / validate_schedule / export_excel

All data is SYNTHETIC. Every number in the CONFIGURATION block is a team
assumption unless a source is given, and can be changed in one place.
"""

import io
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pulp

# =============================================================================
# CONFIGURATION (team assumptions, edit here)
# =============================================================================

SLOT = 0.5                      # half-hour planning slots
DAY_START, DAY_END = 10.0, 24.5   # staff on site 10:00 to 00:30 (customers 11:00 to 00:00)
SLOTS = [DAY_START + i * SLOT for i in range(int((DAY_END - DAY_START) / SLOT))]

SEATS = 50                      # dining seats (team spec)
BAR_STOOLS = 10                 # bar stools, walk-ins only (team spec)

DWELL_HOURS = {"Lunch": 1.5, "Dinner": 2.0, "Bar": 1.0}   # how long a guest stays

# When walk-ins arrive (share of the day's walk-ins arriving in each slot).
WALKIN_ARRIVALS = {
    "Lunch":  {12.0: 0.20, 12.5: 0.30, 13.0: 0.30, 13.5: 0.20},
    "Dinner": {18.0: 0.10, 18.5: 0.15, 19.0: 0.20, 19.5: 0.20, 20.0: 0.15, 20.5: 0.10, 21.0: 0.10},
    "Bar":    {12.0: 0.04, 13.0: 0.04, 14.0: 0.04, 15.0: 0.04, 16.0: 0.04, 17.0: 0.08, 18.0: 0.12,
               19.0: 0.14, 20.0: 0.14, 21.0: 0.12, 22.0: 0.12, 23.0: 0.08},
}

# Productivity: guests in the building that one member of staff can look after.
COVERS_PER_LINE_CHEF = 20
COVERS_PER_KP = 35
COVERS_PER_SERVER = 12
PIZZA_OVEN_FROM_COVERS = 12      # open the pizza station once this many diners are in
HOST_FROM_COVERS = 25            # staff the host stand once this many diners are in
BAR_GUESTS_PER_BARTENDER = 8
DINERS_PER_BARTENDER = 40        # bartenders also make drinks for the dining room

# Cost weights in the objective (GBP).
PENALTY_BOOKED_SHORTFALL = 10_000  # per missing staff-slot against BOOKINGS: effectively never allowed
PENALTY_WALKIN_SHORTFALL = 40      # per missing staff-slot against forecast walk-ins
FULL_TIME_GUARANTEED_HOURS = 35   # per 7-day week, pro rata for part weeks (team assumption)
PENALTY_DISLIKED_DAY = 8           # per shift on a day the person would rather not work
BONUS_PREFERRED_BLOCK = 2          # per shift in the person's preferred block

# Legal rules (Working Time Regulations 1998, GOV.UK). See the Rules sheet.
MIN_REST_HOURS = 11
MAX_DAYS_PER_WEEK = 6
BREAK_REQUIRED_OVER_HOURS = 6
NATIONAL_LIVING_WAGE = 12.71       # GOV.UK, age 21+, from 1 April 2026


@dataclass(frozen=True)
class Shift:
    code: str
    label: str
    block: str                     # availability block it falls in (AM / PM)
    start: float
    end: float
    break_start: Optional[float] = None   # 30-min unpaid break

    @property
    def paid_hours(self):
        return self.end - self.start - (SLOT if self.break_start is not None else 0)

    def on_floor(self, slot):
        return self.start <= slot < self.end and slot != self.break_start


# Shift patterns a person can be given. A long PM shift has four versions that
# differ only in when the break is taken, so the optimiser places breaks in quiet
# slots where cover is still met.
SHIFTS = [
    Shift("AM_FULL", "Day", "AM", 10.0, 16.0),
    Shift("AM_LUNCH", "Lunch", "AM", 11.0, 15.0),
    Shift("PM_DINNER", "Dinner", "PM", 17.0, 23.0),
    Shift("PM_LATE", "Late", "PM", 19.0, 24.5),
    Shift("PM_FULL_B1", "Evening", "PM", 16.0, 24.5, break_start=17.0),
    Shift("PM_FULL_B2", "Evening", "PM", 16.0, 24.5, break_start=17.5),
    Shift("PM_FULL_B3", "Evening", "PM", 16.0, 24.5, break_start=22.5),
    Shift("PM_FULL_B4", "Evening", "PM", 16.0, 24.5, break_start=23.0),
]
SHIFT_BY_CODE = {s.code: s for s in SHIFTS}

# Where in the restaurant people are placed, and who may work there.
STATIONS = {
    # Team assumption: a senior server (Head / Senior Waiter) may act as duty manager.
    "Duty Manager (Floor)": {"area": "Front of House", "roles": {"Manager"}, "senior_only": False,
                             "also": {("Server", "Senior")}},
    "Host Stand":           {"area": "Front of House", "roles": {"Host"}, "senior_only": False},
    "Dining Room":          {"area": "Front of House", "roles": {"Server"}, "senior_only": False},
    "Bar":                  {"area": "Bar", "roles": {"Bartender"}, "senior_only": False},
    "Pass":                 {"area": "Kitchen", "roles": {"Chef"}, "senior_only": True},
    "Hot Line":             {"area": "Kitchen", "roles": {"Chef"}, "senior_only": False},
    "Pizza Oven":           {"area": "Kitchen", "roles": {"Chef"}, "senior_only": False},
    "Dish & Prep":          {"area": "Kitchen", "roles": {"Kitchen Porter"}, "senior_only": False},
}


# =============================================================================
# Helpers
# =============================================================================

def hhmm(hours):
    """24.5 -> '00:30', 17.0 -> '17:00'."""
    total = int(round(hours * 60)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def to_hours(text):
    """'19:30' -> 19.5"""
    h, m = str(text).strip()[:5].split(":")
    return int(h) + int(m) / 60


def week_of(d):
    return d.isocalendar()[1]


# =============================================================================
# 1-3. Load inputs
# =============================================================================

# Accepted spellings for category columns (compared in lower case, spaces trimmed).
ROLE_ALIASES = {
    "manager": "Manager", "general manager": "Manager", "assistant manager": "Manager",
    "supervisor": "Manager", "floor supervisor": "Manager", "duty manager": "Manager",
    "chef": "Chef", "cook": "Chef",
    "kitchen porter": "Kitchen Porter", "kp": "Kitchen Porter", "porter": "Kitchen Porter",
    "dishwasher": "Kitchen Porter", "kitchen-porter": "Kitchen Porter",
    "bartender": "Bartender", "barman": "Bartender", "barmaid": "Bartender", "bar staff": "Bartender",
    "server": "Server", "waiter": "Server", "waitress": "Server", "waiting staff": "Server",
    "host": "Host", "hostess": "Host",
}
SENIORITY_ALIASES = {"senior": "Senior", "snr": "Senior", "sr": "Senior",
                     "junior": "Junior", "jnr": "Junior", "jr": "Junior"}
CONTRACT_ALIASES = {"full-time": "Full-time", "full time": "Full-time", "fulltime": "Full-time", "ft": "Full-time",
                    "part-time": "Part-time", "part time": "Part-time", "parttime": "Part-time", "pt": "Part-time",
                    "student": "Student"}
BLOCK_ALIASES = {"am": "AM", "morning": "AM", "mornings": "AM", "day": "AM", "lunch": "AM",
                 "pm": "PM", "evening": "PM", "evenings": "PM", "night": "PM", "dinner": "PM"}
YES_NO = {"1": 1, "1.0": 1, "yes": 1, "y": 1, "true": 1, "available": 1,
          "0": 0, "0.0": 0, "no": 0, "n": 0, "false": 0, "unavailable": 0}
PREFERENCE_ALIASES = {"1": 1, "+1": 1, "1.0": 1, "prefer": 1, "preferred": 1, "yes": 1,
                      "0": 0, "0.0": 0, "neutral": 0,
                      "-1": -1, "-1.0": -1, "avoid": -1, "rather not": -1, "no": -1}


class DataProblems(ValueError):
    """Raised with EVERY problem found in an input file, one per line."""

    def __init__(self, problems):
        self.problems = problems
        shown = problems[:25]
        more = f"\n- ...and {len(problems) - 25} more" if len(problems) > 25 else ""
        super().__init__("\n".join(f"- {p}" for p in shown) + more)


def _text(v):
    return "" if pd.isna(v) else str(v).strip()


def _number(v):
    """15, '15', '£15.50', ' 15 ' -> float; anything else -> None."""
    if isinstance(v, (int, float)) and not pd.isna(v):
        return float(v)
    t = _text(v).replace("£", "").replace(",", "").replace("GBP", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def _category(v, aliases):
    return aliases.get(_text(v).lower())


def load_staff(path):
    """Read the Staff sheet. Fixes harmless variations (case, spaces, common aliases),
    records warnings in staff.attrs['warnings'], and raises DataProblems listing every
    real problem with its Excel row number."""
    raw = pd.read_excel(path, sheet_name="Staff")
    needed = ["staff_id", "first_name", "job_title", "role", "seniority", "contract",
              "hourly_wage_gbp", "max_weekly_hours", "preferred_block"]
    missing_cols = [c for c in needed if c not in raw.columns]
    if missing_cols:
        raise DataProblems([f"Staff sheet is missing column(s): {', '.join(missing_cols)}"])

    problems, warnings, rows = [], [], []
    for i, r in raw.iterrows():
        sid = _text(r["staff_id"]).upper()
        if not sid.startswith("S") or not sid[1:].isdigit():
            continue                                 # notes and blank rows below the table
        where = f"Staff row {i + 2} ({sid})"
        name = _text(r["first_name"])
        if not name:
            warnings.append(f"{where}: first_name is blank, so the staff ID is used as the name")
            name = sid
        role = _category(r["role"], ROLE_ALIASES)
        if role is None:
            problems.append(f"{where}: role '{_text(r['role'])}' is not recognised "
                            f"(use Manager, Chef, Kitchen Porter, Bartender, Server or Host)")
        elif _text(r["role"]) != role:
            warnings.append(f"{where}: role '{_text(r['role'])}' read as '{role}'")
        seniority = _category(r["seniority"], SENIORITY_ALIASES)
        if seniority is None:
            problems.append(f"{where}: seniority '{_text(r['seniority'])}' must be Senior or Junior")
        contract = _category(r["contract"], CONTRACT_ALIASES)
        if contract is None:
            problems.append(f"{where}: contract '{_text(r['contract'])}' must be Full-time, Part-time or Student")
        wage = _number(r["hourly_wage_gbp"])
        if wage is None or wage <= 0:
            problems.append(f"{where}: hourly_wage_gbp '{_text(r['hourly_wage_gbp'])}' must be a number, e.g. 14.50")
        elif wage < NATIONAL_LIVING_WAGE:
            warnings.append(f"{where}: wage GBP {wage:.2f} is below the National Living Wage "
                            f"(GBP {NATIONAL_LIVING_WAGE})")
        hours = _number(r["max_weekly_hours"])
        if hours is None or not 0 < hours <= 48:
            problems.append(f"{where}: max_weekly_hours '{_text(r['max_weekly_hours'])}' must be a "
                            f"number between 1 and 48")
        block = _text(r["preferred_block"])
        block = BLOCK_ALIASES.get(block.lower(), "Either" if block.lower() in ("either", "any", "") else block)
        rows.append({"staff_id": sid, "first_name": name, "job_title": _text(r["job_title"]) or role,
                     "role": role, "seniority": seniority, "contract": contract,
                     "hourly_wage_gbp": wage, "max_weekly_hours": hours, "preferred_block": block})

    staff = pd.DataFrame(rows, columns=needed)
    dupes = staff["staff_id"][staff["staff_id"].duplicated()].unique().tolist()
    if dupes:
        problems.append(f"Staff sheet: staff_id used more than once: {', '.join(dupes)}")
    if staff.empty:
        problems.append("Staff sheet: no staff rows found (staff_id must look like S01)")
    if problems:
        raise DataProblems(problems)
    staff.attrs["warnings"] = warnings
    return staff


def load_availability(path):
    """Read the Availability sheet with the same approach as load_staff.

    Blank availability is treated as unavailable (0) and a blank or unknown preference
    as neutral (0), both with a warning, because guessing 'available' could put someone
    on shift who cannot work."""
    raw = pd.read_excel(path, sheet_name="Availability")
    needed = ["staff_id", "date", "block", "available", "preference"]
    missing_cols = [c for c in needed if c not in raw.columns]
    if missing_cols:
        raise DataProblems([f"Availability sheet is missing column(s): {', '.join(missing_cols)}"])

    problems, warnings, rows = [], [], []
    blank_avail, odd_pref = [], []
    dates = pd.to_datetime(raw["date"], errors="coerce", dayfirst=True)
    for i, r in raw.iterrows():
        sid = _text(r["staff_id"]).upper()
        if not sid:
            continue
        where = f"Availability row {i + 2} ({sid})"
        d = dates.iloc[i]
        if pd.isna(d):
            problems.append(f"{where}: date '{_text(r['date'])}' is not a date (use e.g. 2026-10-05)")
        block = BLOCK_ALIASES.get(_text(r["block"]).lower())
        if block is None:
            problems.append(f"{where}: block '{_text(r['block'])}' must be AM or PM")
        a = YES_NO.get(_text(r["available"]).lower())
        if _text(r["available"]) == "":
            blank_avail.append(i + 2)
            a = 0
        elif a is None:
            problems.append(f"{where}: available '{_text(r['available'])}' must be 1 or 0 (yes / no)")
        pref = PREFERENCE_ALIASES.get(_text(r["preference"]).lower(), None)
        if pref is None:
            if _text(r["preference"]) and a == 1:
                odd_pref.append(f"{i + 2} ('{_text(r['preference'])}')")
            pref = 0
        rows.append({"staff_id": sid, "date": None if pd.isna(d) else d.date(), "block": block,
                     "available": a, "preference": pref})
    if blank_avail:
        warnings.append(f"Availability: {len(blank_avail)} blank 'available' cell(s) treated as "
                        f"unavailable (rows {', '.join(map(str, blank_avail[:10]))}"
                        f"{'...' if len(blank_avail) > 10 else ''})")
    if odd_pref:
        warnings.append(f"Availability: unrecognised preference treated as neutral in row(s) "
                        f"{', '.join(odd_pref[:10])}{'...' if len(odd_pref) > 10 else ''}")
    if problems:
        raise DataProblems(problems)
    av = pd.DataFrame(rows, columns=needed)
    dupes = av.duplicated(["staff_id", "date", "block"], keep="first")
    if dupes.any():
        warnings.append(f"Availability: {int(dupes.sum())} duplicate row(s) for the same person, "
                        f"date and block; the first one is used")
        av = av[~dupes]
    av = av.reset_index(drop=True)
    av.attrs["warnings"] = warnings
    return av


def load_bookings(path):
    """Confirmed future bookings. Cancelled / no-show rows already have 0 covers."""
    b = pd.read_excel(path, sheet_name="Bookings")
    b = b[b["Covers (confirmed)"] > 0].copy()
    out = pd.DataFrame({
        "booking_id": b["Booking ID"],
        "date": pd.to_datetime(b["Reservation Date"]).dt.date,
        "meal": b["Meal Period"],
        "arrival": b["Reservation Time"].map(to_hours),
        "covers": b["Covers (confirmed)"].astype(int),
        "event": b["Event Type"],
    })
    return out.reset_index(drop=True)


def load_walkin_history(path):
    """Average walk-ins per weekday from October 2025 (lunch, dinner and bar)."""
    h = pd.read_excel(path, sheet_name="Daily Demand")
    means = h.groupby("Day").agg(
        walkin_lunch=("Walk-in Lunch Customers", "mean"),
        walkin_dinner=("Walk-in Dinner Customers", "mean"),
        bar=("Bar Walk-in Customers", "mean"),
        days_observed=("Date", "count"),
    )
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return means.reindex(order).round(1)


# =============================================================================
# 4. Demand: guests in the building per half-hour slot
# =============================================================================

def in_house(arrivals, dwell):
    """arrivals: list of (arrival_hour, people). Returns {slot: people present}."""
    occ = {s: 0.0 for s in SLOTS}
    for arrive, people in arrivals:
        for s in SLOTS:
            overlap = min(arrive + dwell, s + SLOT) - max(arrive, s)
            if overlap > 0:
                occ[s] += people * overlap / SLOT
    return occ


def build_demand(bookings, walkin_means, dates, walkin_multiplier=1.0):
    """One row per date x slot.

    booked_food   diners from confirmed bookings (never capped: we must serve them)
    walkin_food   forecast walk-in diners, capped by the seats left after bookings
    bar           forecast bar guests, capped by bar stools
    """
    rows = []
    for d in dates:
        day = bookings[bookings["date"] == d]
        booked = defaultdict(float)
        for meal in ("Lunch", "Dinner"):
            m = day[day["meal"] == meal]
            occ = in_house(list(zip(m["arrival"], m["covers"])), DWELL_HOURS[meal])
            for s, v in occ.items():
                booked[s] += v

        wk = walkin_means.loc[d.strftime("%A")]
        walk = defaultdict(float)
        for meal, col in (("Lunch", "walkin_lunch"), ("Dinner", "walkin_dinner")):
            total = wk[col] * walkin_multiplier
            arrivals = [(t, total * w) for t, w in WALKIN_ARRIVALS[meal].items()]
            for s, v in in_house(arrivals, DWELL_HOURS[meal]).items():
                walk[s] += v
        bar_total = wk["bar"] * walkin_multiplier
        bar_occ = in_house([(t, bar_total * w) for t, w in WALKIN_ARRIVALS["Bar"].items()],
                           DWELL_HOURS["Bar"])

        for s in SLOTS:
            walk_capped = min(walk[s], max(0.0, SEATS - booked[s]))
            rows.append({
                "date": d, "weekday": d.strftime("%a"), "slot": s, "time": hhmm(s),
                "booked_food": round(booked[s], 2),
                "walkin_food": round(walk_capped, 2),
                "food_total": round(booked[s] + walk_capped, 2),
                "seated_for_staffing": round(min(booked[s] + walk_capped, SEATS), 2),
                "bar": round(min(bar_occ[s], BAR_STOOLS), 2),
                "overbooked": booked[s] > SEATS,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 5. Requirements: staff needed per station per slot
# =============================================================================

def station_requirements(food, bar, slot, per_server=None, per_chef=None):
    """Staff needed at each station in one slot, given guests in the building.

    per_server / per_chef override COVERS_PER_SERVER / COVERS_PER_LINE_CHEF for one
    run without changing the module settings (safe when several people use the app).
    """
    per_server = per_server or COVERS_PER_SERVER
    per_chef = per_chef or COVERS_PER_LINE_CHEF
    kitchen_open = slot < 23.0          # prep from 10:00, last food out by 23:00
    customers_in = 11.0 <= slot < 24.0
    req = {
        "Duty Manager (Floor)": 1,
        "Pass": 1 if kitchen_open else 0,
        "Hot Line": max(1, math.ceil(food / per_chef)) if kitchen_open else 0,
        "Pizza Oven": 1 if kitchen_open and food >= PIZZA_OVEN_FROM_COVERS else 0,
        "Dish & Prep": max(1, math.ceil(food / COVERS_PER_KP)) if slot < 23.5 else 0,
        "Dining Room": (max(1, math.ceil(food / per_server)) if customers_in
                        else (1 if slot < 11.0 else 0)),
        "Host Stand": 1 if food >= HOST_FROM_COVERS else 0,
        "Bar": (max(1, math.ceil(bar / BAR_GUESTS_PER_BARTENDER + food / DINERS_PER_BARTENDER))
                if slot >= 10.5 else 0),
    }
    return req


def build_requirements(demand, per_server=None, per_chef=None):
    rows = []
    for r in demand.itertuples(index=False):
        # Staff for at most SEATS diners: bookings above capacity are flagged in
        # demand["overbooked"] for the manager rather than staffed.
        req_min = station_requirements(min(r.booked_food, SEATS), 0.0, r.slot, per_server, per_chef)
        req_full = station_requirements(r.seated_for_staffing, r.bar, r.slot, per_server, per_chef)
        for station in STATIONS:
            rows.append({
                "date": r.date, "slot": r.slot, "time": r.time, "station": station,
                "req_min": req_min[station],
                "req_full": max(req_full[station], req_min[station]),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 6. Optimisation
# =============================================================================

def eligible_stations(person):
    out = []
    for name, rule in STATIONS.items():
        role_ok = person["role"] in rule["roles"] and (not rule["senior_only"] or person["seniority"] == "Senior")
        if role_ok or (person["role"], person["seniority"]) in rule.get("also", set()):
            out.append(name)
    return out


def rest_conflicts(min_rest=MIN_REST_HOURS):
    """Pairs (shift today, shift tomorrow) that leave less than the legal rest."""
    return [(a.code, b.code) for a in SHIFTS for b in SHIFTS
            if (24 + b.start) - a.end < min_rest]


def build_schedule(staff, availability, requirements, time_limit=45, msg=False):
    """Solve the rota one week (Mon-Sun) at a time, as a manager would.

    Weekly hours and days-off rules sit inside a week, so the only link between
    weeks is the 11-hour rest rule, which is carried over from the last day of
    the previous week. Returns (schedule DataFrame, solver info dict).
    """
    all_dates = sorted(requirements["date"].unique())
    weeks = defaultdict(list)
    for d in all_dates:
        weeks[week_of(d)].append(d)
    parts, infos, previous = [], [], {}
    for wk in sorted(weeks):
        dates = weeks[wk]
        week_req = requirements[requirements["date"].isin(dates)]
        sched, info = _solve_week(staff, availability, week_req, dates, previous, time_limit, msg)
        info["week"] = f"{dates[0]:%d %b}-{dates[-1]:%d %b}"
        parts.append(sched)
        infos.append(info)
        last = sched[sched["date"] == dates[-1]]
        previous = dict(zip(last["staff_id"], last["shift_code"]))
    schedule = pd.concat(parts, ignore_index=True)
    weekly = pd.DataFrame(infos)
    info = {
        "status": ("Optimal" if (weekly["status"] == "Optimal").all()
                   else "Some weeks stopped at the time limit (see weeks table)"),
        "variables": int(weekly["variables"].sum()),
        "constraints": int(weekly["constraints"].sum()),
        "booked_shortfall_staff_slots": round(weekly["booked_shortfall_staff_slots"].sum(), 2),
        "walkin_shortfall_staff_slots": round(weekly["walkin_shortfall_staff_slots"].sum(), 2),
        "weeks": weekly[["week", "status", "variables", "seconds"]],
    }
    return schedule, info


def _solve_week(staff, availability, requirements, dates, previous, time_limit, msg):
    """One week's integer programme. previous = {staff_id: shift_code} worked the day before."""
    import time as _time
    t0 = _time.time()
    persons = {r.staff_id: r._asdict() for r in staff.itertuples(index=False)}
    av = availability.set_index(["staff_id", "date", "block"])

    prob = pulp.LpProblem("Shiftr", pulp.LpMinimize)
    x = {}
    cost_terms = []
    for sid, p in persons.items():
        stations = eligible_stations(p)
        for d in dates:
            for sh in SHIFTS:
                key = (sid, d, sh.block)
                if key not in av.index or av.loc[key, "available"] != 1:
                    continue
                pref = av.loc[key, "preference"]
                for st in stations:
                    v = pulp.LpVariable(f"x_{sid}_{d:%m%d}_{sh.code}_{len(x)}", cat="Binary")
                    x[(sid, d, sh.code, st)] = v
                    c = p["hourly_wage_gbp"] * sh.paid_hours
                    c += PENALTY_DISLIKED_DAY if pref == -1 else 0
                    c -= BONUS_PREFERRED_BLOCK if pref == 1 else 0
                    cost_terms.append(c * v)

    by_person_day = defaultdict(list)
    by_day_station_shift = defaultdict(list)
    for (sid, d, code, st), v in x.items():
        by_person_day[(sid, d)].append((code, v))
        by_day_station_shift[(d, st, code)].append(v)

    # Cover requirements (two tiers, with penalised slack so the model never "fails").
    slack_min, slack_full = {}, {}
    for r in requirements.itertuples(index=False):
        if r.req_full == 0:
            continue
        cover = pulp.lpSum(v for sh in SHIFTS if sh.on_floor(r.slot)
                           for v in by_day_station_shift[(r.date, r.station, sh.code)])
        k = (r.date, r.station, r.slot)
        if r.req_min > 0:
            slack_min[k] = pulp.LpVariable(f"smin_{len(slack_min)}", lowBound=0)
            prob += cover + slack_min[k] >= r.req_min
        slack_full[k] = pulp.LpVariable(f"sfull_{len(slack_full)}", lowBound=0)
        prob += cover + slack_full[k] >= r.req_full

    # One shift per person per day.
    for vs in by_person_day.values():
        prob += pulp.lpSum(v for _, v in vs) <= 1

    # Weekly hours cap and at least one day off per week.
    for sid, p in persons.items():
        weeks = defaultdict(list)
        for d in dates:
            weeks[week_of(d)].append(d)
        for wk_dates in weeks.values():
            vs = [(code, v) for d in wk_dates for code, v in by_person_day.get((sid, d), [])]
            if not vs:
                continue
            hours = pulp.lpSum(SHIFT_BY_CODE[c].paid_hours * v for c, v in vs)
            prob += hours <= p["max_weekly_hours"]
            if p["contract"] == "Full-time":
                # Contracted hours are paid whether used or not, so unused hours
                # cost the same as worked ones.
                guaranteed = FULL_TIME_GUARANTEED_HOURS * len(wk_dates) / 7
                unused = pulp.LpVariable(f"unused_{sid}_{len(cost_terms)}", lowBound=0)
                prob += hours + unused >= guaranteed
                cost_terms.append(p["hourly_wage_gbp"] * unused)
            if len(wk_dates) > MAX_DAYS_PER_WEEK:
                prob += pulp.lpSum(v for _, v in vs) <= MAX_DAYS_PER_WEEK

    # 11 hours' rest between consecutive days (including from last week's final day).
    conflicts = rest_conflicts()
    for sid, code in previous.items():
        first = dict(_sum_by_code(by_person_day.get((sid, dates[0]), [])))
        for a, b in conflicts:
            if a == code and b in first:
                prob += first[b] == 0
    for sid in persons:
        for d in dates:
            nxt = d + timedelta(days=1)
            today = dict(_sum_by_code(by_person_day.get((sid, d), [])))
            tomorrow = dict(_sum_by_code(by_person_day.get((sid, nxt), [])))
            for a, b in conflicts:
                if a in today and b in tomorrow:
                    prob += today[a] + tomorrow[b] <= 1

    prob += (pulp.lpSum(cost_terms)
             + PENALTY_BOOKED_SHORTFALL * pulp.lpSum(slack_min.values())
             + PENALTY_WALKIN_SHORTFALL * pulp.lpSum(slack_full.values()))

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=0.005)
    prob.solve(solver)
    if prob.sol_status == pulp.LpSolutionOptimal:
        status = "Optimal"
    elif prob.sol_status == pulp.LpSolutionIntegerFeasible:
        status = "Feasible (time limit, may not be cheapest)"
    else:
        raise RuntimeError(f"Solver did not return a rota (status: {pulp.LpStatus[prob.status]})")

    rows = []
    for (sid, d, code, st), v in x.items():
        if v.value() and v.value() > 0.5:
            p, sh = persons[sid], SHIFT_BY_CODE[code]
            pref = av.loc[(sid, d, sh.block), "preference"]
            rows.append({
                "date": d, "day": d.strftime("%a"), "staff_id": sid, "employee": p["first_name"],
                "job_title": p["job_title"], "role": p["role"],
                "area": STATIONS[st]["area"], "station": st, "shift": sh.label, "shift_code": code,
                "start": hhmm(sh.start), "end": hhmm(sh.end),
                "break": (f"{hhmm(sh.break_start)}-{hhmm(sh.break_start + SLOT)} (unpaid)"
                          if sh.break_start is not None else "None (shift is 6 h or less)"),
                "paid_hours": sh.paid_hours, "hourly_wage_gbp": p["hourly_wage_gbp"],
                "shift_cost_gbp": round(sh.paid_hours * p["hourly_wage_gbp"], 2),
                "preference": {1: "Preferred", 0: "Neutral", -1: "Would rather not"}[int(pref)],
                "_start": sh.start,
            })
    schedule = (pd.DataFrame(rows)
                .sort_values(["date", "_start", "area", "station", "employee"])
                .drop(columns="_start").reset_index(drop=True))
    schedule = _label_sections(schedule)

    info = {
        "status": status,
        "objective": pulp.value(prob.objective),
        "variables": len(x),
        "constraints": len(prob.constraints),
        "booked_shortfall_staff_slots": round(sum(v.value() or 0 for v in slack_min.values()), 2),
        "walkin_shortfall_staff_slots": round(sum(v.value() or 0 for v in slack_full.values()), 2),
        "seconds": round(_time.time() - t0, 1),
    }
    return schedule, info


def _sum_by_code(pairs):
    acc = defaultdict(list)
    for code, v in pairs:
        acc[code].append(v)
    return [(code, pulp.lpSum(vs)) for code, vs in acc.items()]


def _label_sections(schedule):
    """Give each server on a given day a dining-room section (A, B, C, ...)."""
    schedule = schedule.copy()
    schedule["placement"] = schedule["station"]
    for d, grp in schedule[schedule["station"] == "Dining Room"].groupby("date"):
        for i, idx in enumerate(grp.index):
            schedule.loc[idx, "placement"] = f"Dining Room - Section {chr(ord('A') + i % 6)}"
    return schedule


# =============================================================================
# 7. Reporting and checks
# =============================================================================

def coverage_report(schedule, requirements):
    """Scheduled staff on the floor vs required, per date x station x slot."""
    on_floor = defaultdict(int)
    for r in schedule.itertuples(index=False):
        sh = SHIFT_BY_CODE[r.shift_code]
        for s in SLOTS:
            if sh.on_floor(s):
                on_floor[(r.date, r.station, s)] += 1
    cov = requirements.copy()
    cov["scheduled"] = [on_floor[(r.date, r.station, r.slot)] for r in cov.itertuples(index=False)]
    cov["short_vs_bookings"] = (cov["req_min"] - cov["scheduled"]).clip(lower=0)
    cov["short_vs_forecast"] = (cov["req_full"] - cov["scheduled"]).clip(lower=0)
    return cov


def extra_cover_needed(cov, column="short_vs_bookings"):
    """Group shortfalls into readable blocks, e.g. 'Fri 02 Oct, Hot Line, 19:00-21:30, +1'."""
    rows = []
    short = cov[cov[column] > 0].sort_values(["date", "station", "slot"])
    for (d, st), g in short.groupby(["date", "station"], sort=False):
        run = None
        for r in g.itertuples(index=False):
            if run and r.slot == run["_end"] and getattr(r, column) == run["extra_staff"]:
                run["_end"] = r.slot + SLOT
                continue
            if run:
                rows.append(run)
            run = {"date": d, "day": d.strftime("%a"), "station": st, "from": r.time,
                   "_end": r.slot + SLOT, "extra_staff": int(getattr(r, column))}
        rows.append(run)
    out = pd.DataFrame(rows, columns=["date", "day", "station", "from", "_end", "extra_staff"])
    out["to"] = out["_end"].map(hhmm)
    out["staff_hours"] = 0.0
    if len(out):
        out["staff_hours"] = (out["_end"] - out["from"].map(to_hours)) * out["extra_staff"]
    return out[["date", "day", "station", "from", "to", "extra_staff", "staff_hours"]]


def coverage_score(cov):
    """Share of required staff-slots (bookings + walk-ins) actually covered."""
    need = cov["req_full"].sum()
    covered = cov[["req_full", "scheduled"]].min(axis=1).sum()
    return covered / need if need else 1.0


def validate_schedule(schedule, staff, availability, cov):
    """Independent checks of the solver's output. Every row should say PASS."""
    checks = []
    av = availability.set_index(["staff_id", "date", "block"])["available"]
    staff_i = staff.set_index("staff_id")

    blocks = schedule["shift_code"].map(lambda c: SHIFT_BY_CODE[c].block)
    unavailable = [i for i, r in enumerate(schedule.itertuples(index=False))
                   if av.get((r.staff_id, r.date, blocks.iloc[i]), 0) != 1]
    checks.append(("Nobody works a block they are unavailable for", len(unavailable)))

    per_day = schedule.groupby(["staff_id", "date"]).size()
    checks.append(("At most one shift per person per day", int((per_day > 1).sum())))

    s = schedule.assign(week=schedule["date"].map(week_of))
    hours = s.groupby(["staff_id", "week"])["paid_hours"].sum().reset_index()
    hours["cap"] = hours["staff_id"].map(staff_i["max_weekly_hours"])
    checks.append(("Weekly paid hours within each contract cap (all caps <= 48)",
                   int((hours["paid_hours"] > hours["cap"]).sum())))

    days = s.groupby(["staff_id", "week"]).size()
    checks.append(("At least one day off in every week", int((days > MAX_DAYS_PER_WEEK).sum())))

    rest_breaches = 0
    by_person = {k: g.set_index("date")["shift_code"] for k, g in schedule.groupby("staff_id")}
    for sid, shifts in by_person.items():
        for d, code in shifts.items():
            nxt = d + timedelta(days=1)
            if nxt in shifts.index:
                rest = 24 + SHIFT_BY_CODE[shifts[nxt]].start - SHIFT_BY_CODE[code].end
                rest_breaches += rest < MIN_REST_HOURS
    checks.append(("At least 11 h rest between shifts", rest_breaches))

    long_no_break = [c for c in schedule["shift_code"]
                     if SHIFT_BY_CODE[c].end - SHIFT_BY_CODE[c].start > BREAK_REQUIRED_OVER_HOURS
                     and SHIFT_BY_CODE[c].break_start is None]
    checks.append(("Every shift over 6 h includes a break", len(long_no_break)))

    wrong_station = [r for r in schedule.itertuples(index=False)
                     if r.station not in eligible_stations(dict(staff_i.loc[r.staff_id]))]
    checks.append(("Everyone is placed at a station their role allows (Pass = senior chefs)",
                   len(wrong_station)))

    low_wage = staff[staff["hourly_wage_gbp"] < NATIONAL_LIVING_WAGE]
    checks.append((f"All wages >= National Living Wage (GBP {NATIONAL_LIVING_WAGE})", len(low_wage)))

    checks.append(("Demand from confirmed BOOKINGS fully staffed (staff-slots short)",
                   int(cov["short_vs_bookings"].sum())))

    out = pd.DataFrame(checks, columns=["check", "violations"])
    out["result"] = out["violations"].map(lambda n: "PASS" if n == 0 else "FAIL")
    return out


SHIFTR_COLOURS = ("#4F46E5", "#F59E0B")   # Shiftr brand: indigo + amber
DEFAULT_PRINT_COLOURS = (SHIFTR_COLOURS[0], "#F2F2F2")   # header, stripe


def readable_text(hex_colour):
    """'#000000' or '#FFFFFF', whichever is easier to read on hex_colour (WCAG luminance)."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    # contrast with black = (lum + .05) / .05 ; with white = 1.05 / (lum + .05)
    return "#000000" if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else "#FFFFFF"


def tint(hex_colour, amount):
    """Mix hex_colour with white. amount=0 keeps it, amount=1 gives white."""
    h = hex_colour.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02X}" for c in rgb)


def logo_svg(main=None, accent=None, wordmark=True, height=40, text_colour="currentColor",
             r_colour=None):
    """Shiftr logo as SVG: three rota bars, the middle one shifted and highlighted.

    main fills the rounded square, accent fills the shifted bar (and the 'r').
    The other bars use black or white, whichever reads best on main.
    """
    main, accent = main or SHIFTR_COLOURS[0], accent or SHIFTR_COLOURS[1]
    r_colour = r_colour or accent
    bar = readable_text(main)
    mark = (f'<rect width="64" height="64" rx="14" fill="{main}"/>'
            f'<rect x="11" y="15" width="31" height="9" rx="4.5" fill="{bar}"/>'
            f'<rect x="22" y="27.5" width="31" height="9" rx="4.5" fill="{accent}"/>'
            f'<rect x="11" y="40" width="31" height="9" rx="4.5" fill="{bar}"/>')
    if not wordmark:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
                f'width="{height}" height="{height}" role="img" aria-label="Shiftr">{mark}</svg>')
    width = round(height * 232 / 64)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 232 64" width="{width}" '
            f'height="{height}" role="img" aria-label="Shiftr">{mark}'
            f'<text x="78" y="47" font-family="Inter, \'Helvetica Neue\', Arial, sans-serif" '
            f'font-size="42" font-weight="800" letter-spacing="-1" fill="{text_colour}">'
            f'Shift<tspan fill="{r_colour}">r</tspan></text></svg>')


def logo_png(main=None, accent=None, size=128):
    """The logo mark (no wordmark) as a PIL image, e.g. for the browser-tab icon."""
    from PIL import Image, ImageDraw
    main, accent = main or SHIFTR_COLOURS[0], accent or SHIFTR_COLOURS[1]
    k = size / 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=14 * k, fill=main)
    bar = readable_text(main)
    for x, y, colour in ((11, 15, bar), (22, 27.5, accent), (11, 40, bar)):
        d.rounded_rectangle([x * k, y * k, (x + 31) * k, (y + 9) * k], radius=4.5 * k, fill=colour)
    return img


def logo_drawing(main=None, accent=None, height=30):
    """The logo mark plus 'Shiftr' as a ReportLab Drawing for PDFs (vector, no image files)."""
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.lib import colors
    main, accent = main or SHIFTR_COLOURS[0], accent or SHIFTR_COLOURS[1]
    k = height / 64
    d = Drawing(232 * k, height)
    flip = lambda y, h: height - (y + h) * k          # SVG y runs down, PDF y runs up
    d.add(Rect(0, 0, 64 * k, 64 * k, rx=14 * k, ry=14 * k, fillColor=colors.HexColor(main), strokeColor=None))
    bar = readable_text(main)
    for x, y, colour in ((11, 15, bar), (22, 27.5, accent), (11, 40, bar)):
        d.add(Rect(x * k, flip(y, 9), 31 * k, 9 * k, rx=4.5 * k, ry=4.5 * k,
                   fillColor=colors.HexColor(colour), strokeColor=None))
    d.add(String(78 * k, 17 * k, "Shift", fontName="Helvetica-Bold", fontSize=42 * k,
                 fillColor=colors.HexColor("#111111")))
    from reportlab.pdfbase.pdfmetrics import stringWidth
    d.add(String(78 * k + stringWidth("Shift", "Helvetica-Bold", 42 * k), 17 * k, "r",
                 fontName="Helvetica-Bold", fontSize=42 * k, fillColor=colors.HexColor(main)))
    return d


PRINT_COLUMNS = [("date", "Date"), ("employee", "Employee"), ("job_title", "Job title"),
                 ("placement", "Where"), ("start", "Start"), ("end", "End"),
                 ("break", "Break (unpaid)"), ("paid_hours", "Paid hours")]


def timetable_pdf(view, title, subtitle="", colours=None):
    """Printable A4 timetable (PDF bytes) for the rows in `view`.

    Wages and shift costs are deliberately left out: this is the sheet a manager
    hands to staff. If `view` holds one employee, their name moves to the title
    and the employee / job columns are dropped. colours = (main, second) brand
    colours: main fills the header row, a pale tint of second stripes the rows.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    if colours:
        header, stripe = colours[0], tint(colours[1], 0.85)
    else:
        header, stripe = DEFAULT_PRINT_COLOURS
    one_person = view["staff_id"].nunique() == 1 if len(view) else False
    cols = [(c, h) for c, h in PRINT_COLUMNS
            if not (one_person and c in ("employee", "job_title"))]

    def cell(col, value):
        if col == "date":
            return value.strftime("%a %d %b")
        if col == "break":
            return "None" if str(value).startswith("None") else str(value).replace(" (unpaid)", "")
        if col == "paid_hours":
            return f"{value:g}"
        return str(value)

    rows = [[h for _, h in cols]]
    for r in view.sort_values(["date", "start", "employee"]).to_dict("records"):
        rows.append([cell(c, r[c]) for c, _ in cols])

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), title=title,
                            leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    brand_main = colours[0] if colours else SHIFTR_COLOURS[0]
    brand_accent = colours[1] if colours else SHIFTR_COLOURS[1]
    logo = logo_drawing(brand_main, brand_accent, height=26)
    logo.hAlign = "LEFT"
    story = [logo, Spacer(1, 4), Paragraph(title, styles["Title"])]
    if subtitle:
        story.append(Paragraph(subtitle, styles["Normal"]))
    story.append(Spacer(1, 12))
    if len(view):
        table = Table(rows, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(readable_text(header))),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(stripe)]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BFBFBF")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"{len(view)} shifts, {view['paid_hours'].sum():g} paid hours. "
            "Breaks are unpaid and taken away from your station.", styles["Normal"]))
    else:
        story.append(Paragraph("No shifts match the selected filters.", styles["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Generated by Shiftr (prototype, synthetic data).", styles["Italic"]))
    doc.build(story)
    return buf.getvalue()


def rota_grid(schedule):
    """Employee x date view: '16:00-00:30 Bar'."""
    s = schedule.assign(cell=schedule["start"] + "-" + schedule["end"] + " " + schedule["placement"])
    grid = s.pivot_table(index=["staff_id", "employee", "job_title"], columns="date",
                         values="cell", aggfunc="first").fillna("OFF")
    grid.columns = [d.strftime("%a %d %b") for d in grid.columns]
    return grid.reset_index()


def export_excel(path, schedule, grid, demand, cov, checks, staff, info, extra=None):
    """Write the workbook to a file path, or to a file-like object such as io.BytesIO."""
    if isinstance(path, (str, Path)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    hours = (schedule.groupby(["staff_id", "employee", "job_title"])
             .agg(shifts=("date", "count"), paid_hours=("paid_hours", "sum"),
                  cost_gbp=("shift_cost_gbp", "sum")).reset_index())
    short = cov[cov["short_vs_forecast"] > 0]
    summary = pd.DataFrame({
        "item": ["SYNTHETIC DATA - prototype output, not a real rota",
                 "Solver status", "Total labour cost (GBP, base pay excl. tronc)",
                 "Total paid hours", "Shifts scheduled",
                 "Coverage score (bookings + forecast walk-ins)",
                 "Staff-slots short vs bookings (must be 0)",
                 "Staff-slots short vs walk-in forecast"],
        "value": ["", info["status"], round(schedule["shift_cost_gbp"].sum(), 2),
                  schedule["paid_hours"].sum(), len(schedule),
                  f"{coverage_score(cov):.1%}", int(cov["short_vs_bookings"].sum()),
                  int(cov["short_vs_forecast"].sum())],
    })
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        schedule.drop(columns=["shift_code"]).to_excel(xw, sheet_name="Master_Schedule", index=False)
        grid.to_excel(xw, sheet_name="Rota_Grid", index=False)
        hours.to_excel(xw, sheet_name="Staff_Hours", index=False)
        demand.to_excel(xw, sheet_name="Demand_by_Slot", index=False)
        cov.to_excel(xw, sheet_name="Coverage", index=False)
        short.to_excel(xw, sheet_name="Shortfalls", index=False)
        if extra is not None:
            extra.to_excel(xw, sheet_name="Extra_Cover_Needed", index=False)
        if "weeks" in info:
            info["weeks"].to_excel(xw, sheet_name="Solver_Weeks", index=False)
        checks.to_excel(xw, sheet_name="Validation", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2" if ws.title == "Rota_Grid" else "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 45)
    return path
