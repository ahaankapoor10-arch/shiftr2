"""Shiftr assistant: turns a manager's written notes into checked changes to the staff data.

Examples it understands (one instruction per line or sentence):
    Hugo can't work on Friday 9 Oct
    Kofi is off sick 12-14 Oct
    Leila is unavailable on Monday evenings
    Theo can work Saturday morning
    Set Kofi's max hours to 20
    Pay Aisha £15.50 an hour
    Hire Sam as a waiter, part-time, 25 hours, £14 an hour
    Ben has left

This is a rule-based reader, not a generative AI model: it only makes the change types
listed above, shows exactly what it understood, and changes nothing until the manager
applies it. Anything it cannot read confidently is reported back instead of guessed.

Changes are plain dicts (JSON-friendly) so the app can store, undo and cache them.
"""

import difflib
import re
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

import shiftr_core as sc

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_RE = (r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
              r"mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)s?\b")
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
MONTH_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
DAY_RE = r"(\d{1,2})(?:st|nd|rd|th)?"

NEGATIVE = (r"\b(can'?t|cannot|can not|unavailable|not available|isn'?t available|off|sick|ill|"
            r"holiday|away|busy|won'?t|wont|out|leave|annual leave|day off|appointment)\b")
POSITIVE = r"\b(can work|can do|can cover|available|free|is in|will work|happy to work)\b"
LEAVER = (r"\b(has left|left the (team|company|restaurant)|quit|quits|resigned|no longer works|leaving|left|"
          r"fire|fired|sack|sacked|dismiss|dismissed|let go|let (?:\w+ )?go|letting (?:\w+ )?go|terminate|terminated|"
          r"get rid of|gone for good|no longer with us|remove (?:\w+ )?from the (?:team|staff|rota))\b")
PRONOUN = r"\b(he|she|him|her|they|them|his|their|hers)\b"
HIRE = r"\b(hire|hired|hiring|new hire|new starter|recruit|recruited|onboard|taken on|add a new|add new|new (member|staff|employee|joiner))\b"
WAGE_WORDS = r"\b(wage|pay|paid|rate|raise|rise|earn|earns|salary|per hour|an hour|/h|/hr|p/h|ph|pounds?|quid)\b"
HOURS_WORDS = r"\b(max|maximum|hours|hrs|cap|limit)\b"
SWAP = r"\b(swap|switch|exchange|trade)\b"
COVER = r"\b(give|move|reassign|transfer|hand)\b.*\bto\b|\b(covers?|cover for|covering|takes? over|instead of|replace|replaces)\b"
REMOVE = r"\b(take|remove|drop|cancel|unschedule)\b.*\b(off|shift|shifts|rota|schedule)\b"
SCHEDULE_EDITS = {"swap", "reassign", "remove", "add", "extend"}   # change the finished schedule, not the staff data
ADD = r"\b(add|put|schedule|roster|book|bring in|call in|another shift|extra shift)\b"
EXTEND = r"\b(extend|lengthen|stay (?:on|late|later)|come in (?:early|earlier)|longer shift|full shift)\b"
STATION_WORDS = [("duty manager", "Duty Manager (Floor)"), ("hot line", "Hot Line"), ("line", "Hot Line"),
                 ("pass", "Pass"), ("pizza", "Pizza Oven"), ("bar", "Bar"), ("dining", "Dining Room"),
                 ("floor", "Dining Room"), ("host", "Host Stand"), ("dish", "Dish & Prep"),
                 ("prep", "Dish & Prep"), ("pot wash", "Dish & Prep")]
SUGGESTION_REF = r"\b(?:suggestion|option|idea|fix)\s*(?:#|no\.?|number)?\s*(\d+)\b"
APPLY_WORDS = r"\b(apply|do|use|go with|accept|take|pick|choose|yes|ok|okay|let'?s)\b"
WHO_GOT = (r"\bwho (did you|did u|have you|has|is|was|will|got|gets|took|takes|'?s)\b.*\b(give|gave|given|cover|"
           r"covering|take|taking|taken|got|get|assign|assigned|replace|replacing|instead)\b|\bwho (got|gets|took|"
           r"takes|is covering|covers|replaced)\b")
WHAT_TO_DO = (r"\b(what should i do|what can i do|what do you (suggest|recommend)|how (do|can|should) i "
              r"(fix|cover|solve|sort)|suggest(ions)?|recommend(ations)?|options|fix (the )?gaps?|"
              r"cover (the )?gaps?|what are my options|any ideas)\b")

# Job phrases -> (role, seniority, job title). Longest phrases are checked first.
JOB_PHRASES = {
    "head chef": ("Chef", "Senior", "Head Chef"), "sous chef": ("Chef", "Senior", "Sous Chef"),
    "senior chef de partie": ("Chef", "Senior", "Senior Chef de Partie"),
    "chef de partie": ("Chef", "Junior", "Chef de Partie"), "pizza chef": ("Chef", "Junior", "Pizza Chef"),
    "commis chef": ("Chef", "Junior", "Commis Chef"), "commis": ("Chef", "Junior", "Commis Chef"),
    "line cook": ("Chef", "Junior", "Chef de Partie"), "cook": ("Chef", "Junior", "Chef de Partie"),
    "chef": ("Chef", "Junior", "Chef de Partie"),
    "kitchen porter": ("Kitchen Porter", "Junior", "Kitchen Porter"), "kp": ("Kitchen Porter", "Junior", "Kitchen Porter"),
    "porter": ("Kitchen Porter", "Junior", "Kitchen Porter"), "dishwasher": ("Kitchen Porter", "Junior", "Kitchen Porter"),
    "head bartender": ("Bartender", "Senior", "Head Bartender"), "bartender": ("Bartender", "Junior", "Bartender"),
    "barman": ("Bartender", "Junior", "Bartender"), "barmaid": ("Bartender", "Junior", "Bartender"),
    "head waiter": ("Server", "Senior", "Head Waiter"), "senior waiter": ("Server", "Senior", "Senior Waiter"),
    "waiter": ("Server", "Junior", "Waiter"), "waitress": ("Server", "Junior", "Waiter"),
    "server": ("Server", "Junior", "Waiter"),
    "host": ("Host", "Junior", "Host"), "hostess": ("Host", "Junior", "Host"),
    "general manager": ("Manager", "Senior", "General Manager"),
    "assistant manager": ("Manager", "Senior", "Assistant Manager"),
    "floor supervisor": ("Manager", "Senior", "Floor Supervisor"), "supervisor": ("Manager", "Senior", "Floor Supervisor"),
    "manager": ("Manager", "Senior", "Manager"),
}
NOT_NAMES = set(WEEKDAYS) | {"hire", "hired", "add", "new", "senior", "junior", "head", "sous", "part",
                             "full", "time", "student", "as", "a", "an", "the", "and", "on", "to",
                             "please", "we", "i", "our", "starting", "from", "per", "hour", "hours",
                             "weekends", "weekend", "mornings", "evenings", "only", "called", "named"}
NOT_NAMES |= {w for phrase in JOB_PHRASES for w in phrase.split()}
NOT_NAMES |= {"who", "what", "when", "where", "which", "why", "how", "can", "could", "would", "should", "is",
              "are", "was", "were", "does", "do", "did", "will", "give", "take", "swap", "pay", "set", "make",
              "put", "move", "remove", "please", "thanks", "thank", "sorry", "yes", "no", "ok", "okay", "also",
              "today", "tomorrow", "everyone", "anyone", "someone", "nobody", "there", "they", "she", "he",
              "her", "him", "his", "it", "this", "that", "these", "those", "my", "me", "you", "your",
              "shift", "shifts", "rota", "schedule", "bar", "kitchen", "dining", "pass", "sick", "off"}
NOT_NAMES |= set(MONTHS) | {"october", "september", "november", "december", "january", "february",
                            "march", "april", "june", "july", "august"}

HELP = """I can update the staff data from short notes, one per line, for example:
- *Hugo can't work on Friday 9 Oct*
- *Kofi is off sick 12-14 Oct*
- *Leila is unavailable on Monday evenings*
- *Theo can work Saturday morning*
- *Set Kofi's max hours to 20*
- *Pay Aisha £15.50 an hour*
- *Hire Sam as a waiter, part-time, 25 hours, £14 an hour*
- *Ben has left*
- *Swap Francesca and Pedro's schedules* (add a date to swap just that day)
- *Give Francesca's Friday shift to Pedro* / *Pedro covers Francesca on 9 Oct*
- *Take Pedro off Friday 9 Oct*
- *Add Theo to the hot line on Friday 9 Oct evening* / *Extend Aisha's shift on 9 Oct*
- *What should I do about the gaps?* / *Tell me more about suggestion 2* / *Do suggestion 1*

Dates can be like *9 Oct*, *9/10*, *12-14 Oct*, a weekday (*Fridays*), *weekends* or *weekdays*.
Add *morning* or *evening* to change only that shift; otherwise both shifts change.
Nothing changes until you click **Apply**."""


# ---------------------------------------------------------------- helpers

def _split(text):
    parts = re.split(r"[\n;]+|(?<=[.!?])\s+", text)
    return [p.strip(" .,!?-•*") for p in parts if p.strip(" .,!?-•*")]


def _fmt_dates(days, all_dates):
    days = sorted(days)
    if len(days) == len(all_dates):
        return "every day this month"
    shown = ", ".join(d.strftime("%a %d %b") for d in days[:6])
    return shown + (f" and {len(days) - 6} more" if len(days) > 6 else "")


def _blocks(s):
    am = re.search(r"\b(morning|mornings|am|lunch|lunchtime|lunches|daytime|day shift|days)\b", s)
    pm = re.search(r"\b(evening|evenings|night|nights|pm|dinner|dinners|late|lates|close|closing)\b", s)
    if am and not pm:
        return ["AM"]
    if pm and not am:
        return ["PM"]
    return ["AM", "PM"]


def _block_text(blocks):
    return "both shifts" if len(blocks) == 2 else ("morning shift (AM)" if blocks == ["AM"] else "evening shift (PM)")


def find_people(sentence, staff):
    """Staff mentioned by name or ID. Returns (list of staff_ids, note about fuzzy matches)."""
    s = sentence.lower()
    ids = [sid for sid in staff["staff_id"] if re.search(rf"\b{sid.lower()}\b", s)]
    names = {n.lower(): sid for n, sid in zip(staff["first_name"], staff["staff_id"])}
    for n, sid in names.items():
        if re.search(rf"\b{re.escape(n)}\b", s) and sid not in ids:
            ids.append(sid)
    note = ""
    if not ids:
        words = [w for w in re.findall(r"[a-z]{3,}", s) if w not in NOT_NAMES]
        for w in words:
            match = difflib.get_close_matches(w, list(names), n=1, cutoff=0.8)
            if match:
                ids.append(names[match[0]])
                note = f"I read '{w}' as **{match[0].title()}**. "
                break
    return ids, note


def find_dates(sentence, all_dates):
    """Dates mentioned in the sentence, limited to the month being scheduled."""
    s = sentence.lower()
    year = all_dates[0].year
    found = set()

    def add(day, month):
        try:
            d = date(year, month, day)
        except ValueError:
            return
        found.add(d)

    for a, b, m in re.findall(rf"{DAY_RE}\s*(?:-|–|to|until|till|through)\s*{DAY_RE}\s+(?:of\s+)?{MONTH_RE}", s):
        for day in range(int(a), int(b) + 1):
            add(day, MONTHS[m[:3]])
    for a, m1, b, m2 in re.findall(
            rf"{DAY_RE}\s+(?:of\s+)?{MONTH_RE}\s*(?:-|–|to|until|till|through)\s*{DAY_RE}\s+(?:of\s+)?{MONTH_RE}", s):
        start, end = date(year, MONTHS[m1[:3]], int(a)), date(year, MONTHS[m2[:3]], int(b))
        found |= {d for d in all_dates if start <= d <= end}
    for day, m in re.findall(rf"{DAY_RE}\s+(?:of\s+)?{MONTH_RE}", s):
        add(int(day), MONTHS[m[:3]])
    for m, day in re.findall(rf"{MONTH_RE}\s+{DAY_RE}\b", s):
        add(int(day), MONTHS[m[:3]])
    for day, month in re.findall(r"\b(\d{1,2})/(\d{1,2})\b", s):
        add(int(day), int(month))

    if re.search(r"\btoday\b", s):
        found.add(date.today())
    if re.search(r"\btomorrow\b", s):
        found.add(date.today() + timedelta(days=1))
    explicit = {d for d in found if d in all_dates}
    outside = sorted(found - set(all_dates))
    if explicit:
        return sorted(explicit), outside
    weekday_hits = {WEEKDAYS.index(next(w for w in WEEKDAYS if w.startswith(m[:3])))
                    for m in re.findall(WEEKDAY_RE, s)}
    if re.search(r"\bweekends?\b", s):
        weekday_hits |= {5, 6}
    if re.search(r"\bweekdays?\b", s):
        weekday_hits |= {0, 1, 2, 3, 4}
    if weekday_hits:
        return sorted(d for d in all_dates if d.weekday() in weekday_hits), outside
    return [], outside


def _next_id(staff):
    nums = [int(s[1:]) for s in staff["staff_id"] if s[1:].isdigit()]
    return f"S{max(nums, default=0) + 1:02d}"


def _money(s):
    m = re.search(r"(?:£|gbp\s*)(\d+(?:\.\d{1,2})?)", s) or \
        re.search(r"(\d+(?:\.\d{1,2})?)\s*(?:pounds?|quid|gbp)\b", s) or \
        re.search(r"(\d+(?:\.\d{1,2})?)\s*(?:an|per|a|/)\s*(?:hour|hr|h)\b", s) or \
        re.search(r"(?:pay|wage|rate|paid|earn|earns|raise)\D{0,25}?(\d+(?:\.\d{1,2})?)", s)
    return float(m.group(1)) if m else None


def _hours(s):
    m = re.search(r"(\d{1,2}(?:\.\d)?)\s*(?:h|hr|hrs|hours)\b", s) or \
        re.search(r"(?:max|maximum|cap|limit|up to|at most|no more than)\D{0,20}?(\d{1,2}(?:\.\d)?)", s)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------- one sentence -> change(s)

def _hire(sentence, staff, all_dates):
    s = sentence.lower()
    job = next(((k, v) for k, v in sorted(JOB_PHRASES.items(), key=lambda kv: -len(kv[0]))
                if re.search(rf"\b{re.escape(k)}\b", s)), None)
    named = re.search(r"\b(?:called|named)\s+([A-Za-z][a-z'-]+)", sentence)
    candidates = [named.group(1)] if named else [
        w for w in re.findall(r"\b([A-Z][a-z'-]+)\b", sentence) if w.lower() not in NOT_NAMES]
    if not candidates:
        after = re.search(r"\b(?:hire|hired|recruit|add|onboard)\s+([a-z][a-z'-]+)", s)
        if after and after.group(1) not in NOT_NAMES:
            candidates = [after.group(1)]
    if not candidates:
        return None, "I can see you want to hire someone, but not their **name**. Try *Hire Sam as a waiter*."
    if not job:
        return None, (f"What job will **{candidates[0].title()}** do? Try e.g. *Hire {candidates[0].title()} "
                      f"as a waiter* (roles: chef, kitchen porter, bartender, waiter, host, manager).")
    name = candidates[0].title()
    if name.lower() in set(staff["first_name"].str.lower()):
        return None, f"There is already a staff member called **{name}**. Use a different name or their surname initial."
    role, seniority, title = job[1]
    if re.search(r"\b(senior|head|sous)\b", s):
        seniority = "Senior"
    contract = ("Student" if re.search(r"\bstudent\b", s) else
                "Full-time" if re.search(r"\bfull[- ]?time|\bft\b", s) else
                "Part-time" if re.search(r"\bpart[- ]?time|\bpt\b", s) else None)
    hours = _hours(s)
    contract = contract or ("Full-time" if hours and hours >= 35 else "Part-time")
    hours = hours or {"Full-time": 40, "Part-time": 25, "Student": 20}[contract]
    notes = []
    if hours > 48:
        return None, f"{hours:g} hours a week is over the legal 48-hour limit. Please give 48 or fewer."
    if contract == "Student" and hours > 20:
        notes.append("students are usually limited to 20 hours in term time")
    wage = _money(s)
    if wage is None:
        same = staff[(staff["role"] == role) & (staff["seniority"] == seniority)]["hourly_wage_gbp"]
        wage = float(same.median()) if len(same) else 14.0
        notes.append(f"no wage given, so I used £{wage:.2f}/h (the usual rate for this job)")
    if wage < sc.NATIONAL_LIVING_WAGE:
        return None, f"£{wage:.2f}/h is below the National Living Wage (£{sc.NATIONAL_LIVING_WAGE}). Not applied."
    days, _ = find_dates(s, all_dates)
    blocks = _blocks(s)
    if not days:
        days = list(all_dates)
    change = {"type": "hire", "staff_id": _next_id(staff), "first_name": name, "job_title": title,
              "role": role, "seniority": seniority, "contract": contract, "hourly_wage_gbp": wage,
              "max_weekly_hours": hours, "dates": [d.isoformat() for d in days], "blocks": blocks}
    change["summary"] = (f"**Hire {name}** ({change['staff_id']}) as {title} ({seniority}, {contract}), "
                         f"£{wage:.2f}/h, up to {hours:g} h/week, available {_fmt_dates(days, all_dates)}, "
                         f"{_block_text(blocks)}" + (f". Note: {'; '.join(notes)}" if notes else ""))
    return change, None


def interpret(text, staff, all_dates, context=None):
    """Read a message. Returns (changes, replies): proposed changes plus anything to tell the manager.
    `context` (a dict kept by the app) remembers the last people mentioned, so 'her' / 'him' work."""
    context = context if context is not None else {}
    changes, replies = [], []
    all_dates = sorted(all_dates)
    staff = staff.copy()   # grows as hires are read, so later lines can refer to new people
    for sentence in _split(text):
        s = sentence.lower()
        if re.fullmatch(r"(help|\?|what can you do\??|hi|hello|hey)", s):
            replies.append(HELP)
            continue
        if re.search(WHO_GOT, s):
            recent = context.get("recent") or []
            replies.append(("Here's who took the shifts:\n" + "\n".join(f"- {x}" for x in recent)) if recent else
                           "I haven't handed any shifts over yet. Shifts are only reassigned after you click "
                           "**Apply** (or **Run**, if you left them to the optimiser).")
            continue
        suggestions = context.get("suggestions") or []
        ref = re.search(SUGGESTION_REF, s)
        if ref:
            n = int(ref.group(1))
            sug = next((x for x in suggestions if x["id"] == n), None)
            if sug is None:
                replies.append(f"There's no suggestion {n} at the moment."
                               + (" Current suggestions:\n" + list_suggestions(suggestions) if suggestions else
                                  " Every confirmed booking is covered, so there are no suggestions."))
            elif re.search(APPLY_WORDS, s) and not sug["changes"]:
                replies.append(f"Suggestion {n} is advice only, so there's nothing for me to apply: {sug['text']}")
            elif re.search(APPLY_WORDS, s):
                changes += [dict(c, from_suggestion=n) for c in sug["changes"]]
            else:
                replies.append(explain_suggestion(sug))
            continue
        if re.search(WHAT_TO_DO, s) or (re.search(ADD + "|" + EXTEND, s)
                                         and re.search(r"\b(someone|anyone|somebody)\b", s)):
            replies.append(("Here's what I'd suggest for the cover gaps (cheapest first). Say e.g. "
                            "*do suggestion 1* or click Apply in the **Cover gaps** tab:\n" + list_suggestions(suggestions))
                           if suggestions else "Every confirmed booking is covered right now, so there's nothing to fix.")
            continue
        if re.search(HIRE, s):
            change, reply = _hire(sentence, staff, all_dates)
            if change:
                changes.append(change)
                staff = pd.concat([staff, pd.DataFrame([{k: change[k] for k in (
                    "staff_id", "first_name", "job_title", "role", "seniority", "contract",
                    "hourly_wage_gbp", "max_weekly_hours")}])], ignore_index=True)
            replies += [reply] if reply else []
            continue

        people, note = find_people(sentence, staff)
        if not people and re.search(PRONOUN, s) and context.get("last_people"):
            people = [p for p in context["last_people"] if p in set(staff["staff_id"])]
        if people:
            context["last_people"] = people
        if not people:
            unknown = [w for w in re.findall(r"\b([A-Z][a-z'-]+)\b", sentence) if w.lower() not in NOT_NAMES]
            if unknown:
                replies.append(f"There's no one called **{unknown[0]}** in the staff data loaded right now "
                               f"({len(staff)} people). If {unknown[0]} is a new hire, upload the staff file that "
                               f"includes them (e.g. the *v2_new_hires* file) or add them first with "
                               f"*Hire {unknown[0]} as a ...*.")
            else:
                replies.append(f"I couldn't find a staff member in *\"{sentence}\"*. Use their first name "
                               f"(e.g. Kofi) or staff ID (e.g. S38). Type **help** for examples.")
            continue
        name = {r.staff_id: f"{r.first_name} ({r.staff_id})" for r in staff.itertuples()}
        days, outside = find_dates(sentence, all_dates)
        if outside:
            replies.append(f"Ignored date(s) outside this schedule: "
                           f"{', '.join(d.strftime('%d %b') for d in outside)}.")

        if re.search(LEAVER, s) and not re.search(r"\bleave on\b|\bon leave\b|annual leave|\bfir(e|es|ing) up\b", s):
            for sid in people:
                start = days[0] if days else (date.today() if date.today() in all_dates else all_dates[0])
                left = [d for d in all_dates if d >= start]
                changes.append({"type": "availability", "staff_id": sid, "available": 0,
                                "dates": [d.isoformat() for d in left], "blocks": ["AM", "PM"],
                                "summary": f"{note}**{name[sid]} has left**: unavailable from "
                                           f"{start:%a %d %b} to the end of the month"})
            continue

        when = (_fmt_dates(days, all_dates) if days else "the whole period being scheduled")
        if re.search(SWAP, s):
            if len(people) != 2:
                replies.append("To swap, name exactly two people, e.g. *Swap Francesca and Pedro on 9 Oct*.")
                continue
            a, b = _in_order(sentence, staff, people)
            changes.append({"type": "swap", "staff_id": a, "other_id": b,
                            "dates": [d.isoformat() for d in days] or None,
                            "summary": f"{note}**Swap shifts** between **{name[a]}** and **{name[b]}** on {when}"})
            continue
        if re.search(COVER, s) and len(people) == 2:
            giver, taker = _giver_taker(sentence, staff, people)
            changes.append({"type": "reassign", "staff_id": giver, "other_id": taker,
                            "dates": [d.isoformat() for d in days] or None,
                            "summary": f"{note}**Give {name[giver]}'s shift(s)** on {when} **to {name[taker]}**"})
            continue
        if re.search(EXTEND, s) and len(people) == 1 and days:
            for d in days:
                changes.append({"type": "extend", "staff_id": people[0], "dates": [d.isoformat()],
                                "blocks": _blocks(s) if len(_blocks(s)) == 1 else None,
                                "summary": f"{note}**Extend {name[people[0]]}'s shift** on {d:%a %d %b} "
                                           f"to the full {'' if len(_blocks(s)) == 2 else _block_text(_blocks(s)) + ' '}shift"})
            continue
        if re.search(ADD, s) and len(people) == 1 and days and not re.search(REMOVE, s):
            blocks = _blocks(s)
            station = next((st for w, st in STATION_WORDS if re.search(rf"\b{w}\b", s)), None)
            for d in days:
                for b in blocks if len(blocks) == 1 else [None]:
                    changes.append({"type": "add", "staff_id": people[0], "dates": [d.isoformat()],
                                    "block": b, "station": station,
                                    "summary": f"{note}**Add a shift for {name[people[0]]}** on {d:%a %d %b}"
                                               + (f", {'morning' if b == 'AM' else 'evening'}" if b else "")
                                               + (f" at {station}" if station else "")})
            continue
        if re.search(REMOVE, s):
            if not days:
                replies.append(f"Which day should I take {', '.join(name[p] for p in people)} off? "
                               f"e.g. *Take Pedro off Friday 9 Oct*.")
                continue
            for sid in people:
                changes.append({"type": "remove", "staff_id": sid, "dates": [d.isoformat() for d in days],
                                "summary": f"{note}**Remove {name[sid]}'s shift(s)** on {when} "
                                           f"(nobody replaces them)"})
            continue

        wage = _money(s) if re.search(WAGE_WORDS, s) or "£" in s else None
        if wage is not None:
            if wage < sc.NATIONAL_LIVING_WAGE:
                replies.append(f"£{wage:.2f}/h is below the National Living Wage "
                               f"(£{sc.NATIONAL_LIVING_WAGE}), so I haven't changed it.")
                continue
            for sid in people:
                old = float(staff.loc[staff["staff_id"] == sid, "hourly_wage_gbp"].iloc[0])
                if abs(old - wage) < 0.005:
                    replies.append(f"{name[sid]} is already paid £{old:.2f}/h, so nothing to change.")
                    continue
                changes.append({"type": "wage", "staff_id": sid, "hourly_wage_gbp": wage,
                                "summary": f"{note}**{name[sid]}**: wage £{old:.2f} → **£{wage:.2f}**/h"})
            continue

        hours = _hours(s) if re.search(HOURS_WORDS, s) and not days else None
        if hours is not None:
            if not 1 <= hours <= 48:
                replies.append(f"Max weekly hours must be between 1 and 48 (UK Working Time Regulations); "
                               f"you wrote {hours:g}.")
                continue
            for sid in people:
                row = staff[staff["staff_id"] == sid].iloc[0]
                if abs(float(row["max_weekly_hours"]) - hours) < 0.01:
                    replies.append(f"{name[sid]}'s max hours are already {hours:g} h/week, so nothing to change.")
                    continue
                warn = (" (note: students are usually limited to 20 h in term time)"
                        if row["contract"] == "Student" and hours > 20 else "")
                changes.append({"type": "max_hours", "staff_id": sid, "max_weekly_hours": hours,
                                "summary": f"{note}**{name[sid]}**: max hours {row['max_weekly_hours']:g} → "
                                           f"**{hours:g}** h/week{warn}"})
            continue

        neg, pos = re.search(NEGATIVE, s), re.search(POSITIVE, s)
        picked = [] if neg else _suggestion_for(s, people, days, suggestions)
        if len(picked) == 1:
            changes += [dict(c, from_suggestion=picked[0]["id"]) for c in picked[0]["changes"]]
            continue
        if len(picked) > 1:
            replies.append("That matches more than one suggestion. Which one?\n" + "\n".join(
                f"- *do suggestion {x['id']}*: {x['text']}" for x in picked))
            continue
        if not (neg or pos):
            replies.append(f"I found {', '.join(name[p] for p in people)} but not what to change in "
                           f"*\"{sentence}\"*. Type **help** for examples.")
            continue
        if not days:
            replies.append(f"Which day(s)? I couldn't find a date in *\"{sentence}\"* "
                           f"(try *9 Oct*, *12-14 Oct*, *Fridays* or *weekends*).")
            continue
        available = 0 if neg else 1
        blocks = _blocks(s)
        for sid in people:
            changes.append({"type": "availability", "staff_id": sid, "available": available,
                            "dates": [d.isoformat() for d in days], "blocks": blocks,
                            "summary": f"{note}**{name[sid]}**: **{'unavailable' if not available else 'available'}** "
                                       f"on {_fmt_dates(days, all_dates)}, {_block_text(blocks)}"})
    return changes, replies


# ---------------------------------------------------------------- apply

def _in_order(sentence, staff, ids):
    """The given staff IDs in the order their names appear in the sentence."""
    s = sentence.lower()
    first = dict(zip(staff["staff_id"], staff["first_name"].str.lower()))

    def pos(sid):
        hits = [m.start() for m in re.finditer(rf"\b({re.escape(first[sid])}|{sid.lower()})\b", s)]
        return hits[0] if hits else len(s)
    return sorted(ids, key=pos)


def _giver_taker(sentence, staff, ids):
    """Who gives up the shift and who takes it, from phrasing like 'give A's shift to B',
    'B covers A', 'B takes over from A', 'B instead of A'."""
    s = sentence.lower()
    a, b = _in_order(sentence, staff, ids)
    first = dict(zip(staff["staff_id"], staff["first_name"].str.lower()))
    if re.search(rf"\bto\s+({re.escape(first[b])}|{b.lower()})\b", s):
        return a, b                    # give A's shift to B
    if re.search(rf"\bto\s+({re.escape(first[a])}|{a.lower()})\b", s):
        return b, a
    return b, a                        # 'B covers A', 'B takes over from A', 'B instead of A'


def _person_fields(staff, sid):
    row = staff[staff["staff_id"] == sid].iloc[0]
    return {"staff_id": sid, "employee": row["first_name"], "job_title": row["job_title"],
            "role": row["role"], "hourly_wage_gbp": float(row["hourly_wage_gbp"])}


def shift_row(staff, availability, sid, d, code, station):
    """One schedule row in the same format build_schedule produces."""
    p = staff[staff["staff_id"] == sid].iloc[0]
    sh = sc.SHIFT_BY_CODE[code]
    pref = availability.set_index(["staff_id", "date", "block"])["preference"].get((sid, d, sh.block), 0)
    return {"date": d, "day": d.strftime("%a"), "staff_id": sid, "employee": p["first_name"],
            "job_title": p["job_title"], "role": p["role"], "area": sc.STATIONS[station]["area"],
            "station": station, "shift": sh.label, "shift_code": code,
            "start": sc.hhmm(sh.start), "end": sc.hhmm(sh.end),
            "break": (f"{sc.hhmm(sh.break_start)}-{sc.hhmm(sh.break_start + sc.SLOT)} (unpaid)"
                      if sh.break_start is not None else "None (shift is 6 h or less)"),
            "paid_hours": sh.paid_hours, "hourly_wage_gbp": float(p["hourly_wage_gbp"]),
            "shift_cost_gbp": round(sh.paid_hours * float(p["hourly_wage_gbp"]), 2),
            "preference": {1: "Preferred", 0: "Neutral", -1: "Would rather not"}[int(pref)],
            "placement": station}


def _usual_station(schedule, staff, sid):
    mine = schedule[schedule["staff_id"] == sid]["station"]
    if len(mine):
        return mine.mode().iloc[0]
    return sc.eligible_stations(dict(staff.set_index("staff_id").loc[sid]))[0]


def rule_problems(schedule, staff, availability, people):
    """Hard-rule problems for the given people in a schedule (empty list = all fine)."""
    problems = []
    av = availability.set_index(["staff_id", "date", "block"])["available"]
    st = staff.set_index("staff_id")
    for sid in people:
        mine = schedule[schedule["staff_id"] == sid]
        nm = st.loc[sid, "first_name"]
        for r in mine.itertuples(index=False):
            block = sc.SHIFT_BY_CODE[r.shift_code].block
            if av.get((sid, r.date, block), 0) != 1:
                problems.append(f"{nm} is not available on {r.date:%a %d %b} ({block})")
            if r.station not in sc.eligible_stations(dict(st.loc[sid])):
                problems.append(f"{nm} ({st.loc[sid, 'role']}) can't work at {r.station} on {r.date:%a %d %b}")
        per_day = mine.groupby("date").size()
        for d in per_day[per_day > 1].index:
            problems.append(f"{nm} would have two shifts on {d:%a %d %b}")
        if len(mine):
            weeks = mine.assign(week=mine["date"].map(sc.week_of)).groupby("week")
            for _, g in weeks:
                if g["paid_hours"].sum() > st.loc[sid, "max_weekly_hours"]:
                    problems.append(f"{nm} would work {g['paid_hours'].sum():g} h in the week of "
                                    f"{g['date'].min():%d %b}, over their {st.loc[sid, 'max_weekly_hours']:g} h limit")
                if g["date"].nunique() > sc.MAX_DAYS_PER_WEEK:
                    problems.append(f"{nm} would have no day off in the week of {g['date'].min():%d %b}")
            shifts = mine.drop_duplicates("date").set_index("date")["shift_code"]   # doubles reported above
            for d, code in shifts.items():
                nxt = d + timedelta(days=1)
                if nxt in shifts.index:
                    rest = 24 + sc.SHIFT_BY_CODE[shifts[nxt]].start - sc.SHIFT_BY_CODE[code].end
                    if rest < sc.MIN_REST_HOURS:
                        problems.append(f"{nm} would get only {rest:g} h rest before {nxt:%a %d %b}")
    return sorted(set(problems))


def apply_schedule_edit(schedule, staff, availability, edit):
    """Apply one swap / reassign / remove edit. Returns (new schedule, problems).
    If problems is not empty the edit breaks a rule and the original schedule is returned."""
    sched = schedule.copy()
    days = set(sched["date"]) if not edit.get("dates") else {date.fromisoformat(d) for d in edit["dates"]}
    a, b = edit["staff_id"], edit.get("other_id")
    in_scope = sched["date"].isin(days)
    rows_a = sched.index[in_scope & (sched["staff_id"] == a)]
    rows_b = sched.index[in_scope & (sched["staff_id"] == b)] if b else []
    if edit["type"] == "remove":
        if not len(rows_a):
            return schedule, ["they have no shifts on those days"]
        sched = sched.drop(rows_a)
    elif edit["type"] == "add":
        d = date.fromisoformat(edit["dates"][0])
        if len(rows_a):
            return schedule, [f"{_person_fields(staff, a)['employee']} already works on {d:%a %d %b} "
                              f"(ask to extend that shift instead)"]
        station = edit.get("station") or _usual_station(schedule, staff, a)
        code = edit.get("shift_code") or {"AM": "AM_FULL", "PM": "PM_DINNER"}[edit.get("block") or "PM"]
        sched = pd.concat([sched, pd.DataFrame([shift_row(staff, availability, a, d, code, station)])],
                          ignore_index=True)
    elif edit["type"] == "extend":
        d = date.fromisoformat(edit["dates"][0])
        if not len(rows_a):
            return schedule, [f"{_person_fields(staff, a)['employee']} isn't working on {d:%a %d %b} "
                              f"(add a shift instead)"]
        i = rows_a[0]
        old = sc.SHIFT_BY_CODE[sched.loc[i, "shift_code"]]
        code = edit.get("shift_code") or {"AM": "AM_FULL", "PM": "PM_FULL_B4"}[old.block]
        new = sc.SHIFT_BY_CODE[code]
        if new.paid_hours <= old.paid_hours:
            return schedule, [f"that shift is already a full {old.label.lower()} shift"]
        row = shift_row(staff, availability, a, d, code, sched.loc[i, "station"])
        for col, val in row.items():
            sched.at[i, col] = val
    else:
        if edit["type"] == "reassign" and not len(rows_a):
            return schedule, ["the person giving up the shift isn't working on those days"]
        if edit["type"] == "swap" and not len(rows_a) and not len(rows_b):
            return schedule, ["neither person is working on those days"]
        fa, fb = _person_fields(staff, a), _person_fields(staff, b)
        for col, val in fb.items():
            sched.loc[rows_a, col] = val
        if edit["type"] == "swap":
            for col, val in fa.items():
                sched.loc[rows_b, col] = val
        pref = availability.set_index(["staff_id", "date", "block"])["preference"]
        labels = {1: "Preferred", 0: "Neutral", -1: "Would rather not"}
        changed = list(rows_a) + (list(rows_b) if edit["type"] == "swap" else [])
        for i in changed:
            r = sched.loc[i]
            sched.loc[i, "shift_cost_gbp"] = round(r["paid_hours"] * r["hourly_wage_gbp"], 2)
            sched.loc[i, "preference"] = labels[int(pref.get(
                (r["staff_id"], r["date"], sc.SHIFT_BY_CODE[r["shift_code"]].block), 0))]
    # Only people who gain or change shifts can break a rule (giving a shift away can't).
    people = [b] if edit["type"] == "reassign" else [p for p in (a, b) if p]
    if edit["type"] == "remove":
        people = []
    problems = rule_problems(sched, staff, availability, people)
    if problems:
        return schedule, problems
    sched = sched.sort_values(["date", "start", "area", "station", "employee"]).reset_index(drop=True)
    return sc._label_sections(sched.drop(columns=["placement"])), []


def apply_schedule_edit_partly(schedule, staff, availability, edit):
    """Like apply_schedule_edit, but if the whole edit breaks a rule, apply it day by day
    and keep only the days that work. Returns (schedule, edit actually applied or None,
    list of 'day: reason' strings that were skipped)."""
    new, problems = apply_schedule_edit(schedule, staff, availability, edit)
    if not problems:
        return new, edit, []
    if edit["type"] == "remove":
        return schedule, None, problems
    days = sorted(set(schedule["date"])) if not edit.get("dates") else \
        [date.fromisoformat(d) for d in edit["dates"]]
    done, skipped, sched = [], [], schedule
    for d in days:
        one = dict(edit, dates=[d.isoformat()])
        trial, why = apply_schedule_edit(sched, staff, availability, one)
        if why:
            if "isn't working" not in "; ".join(why) and "neither person" not in "; ".join(why):
                skipped.append(f"{d:%a %d %b}: {'; '.join(why)}")
        else:
            sched, done = trial, done + [d]
    if not done:
        return schedule, None, skipped or problems
    applied = dict(edit, dates=[d.isoformat() for d in done],
                   summary=edit["summary"].split(" on ")[0] + f" on {', '.join(d.strftime('%a %d %b') for d in done)}")
    return sched, applied, skipped


def apply_changes(staff, availability, changes):
    """Return copies of staff and availability with the changes applied, in order.
    Schedule edits (swap / reassign / remove) are skipped here; see apply_schedule_edit."""
    staff = staff.copy()
    av = availability.copy()
    all_dates = sorted(av["date"].dropna().unique())
    for c in changes:
        t, sid = c["type"], c["staff_id"]
        if t in SCHEDULE_EDITS:
            continue
        if t == "availability":
            days = {date.fromisoformat(d) for d in c["dates"]}
            mask = (av["staff_id"] == sid) & av["date"].isin(days) & av["block"].isin(c["blocks"])
            av.loc[mask, "available"] = c["available"]
        elif t == "wage":
            staff.loc[staff["staff_id"] == sid, "hourly_wage_gbp"] = c["hourly_wage_gbp"]
        elif t == "max_hours":
            staff.loc[staff["staff_id"] == sid, "max_weekly_hours"] = c["max_weekly_hours"]
        elif t == "hire" and sid not in set(staff["staff_id"]):
            staff = pd.concat([staff, pd.DataFrame([{
                "staff_id": sid, "first_name": c["first_name"], "job_title": c["job_title"],
                "role": c["role"], "seniority": c["seniority"], "contract": c["contract"],
                "hourly_wage_gbp": c["hourly_wage_gbp"], "max_weekly_hours": c["max_weekly_hours"],
                "preferred_block": c["blocks"][0] if len(c["blocks"]) == 1 else "Either"}])],
                ignore_index=True)
            days = {date.fromisoformat(d) for d in c["dates"]}
            av = pd.concat([av, pd.DataFrame([
                {"staff_id": sid, "date": d, "block": b,
                 "available": int(d in days and b in c["blocks"]), "preference": 0}
                for d in all_dates for b in ("AM", "PM")])], ignore_index=True)
    return staff, av


# ---------------------------------------------------------------- suggestions for cover gaps

def gap_blocks(coverage):
    """Runs of half-hour slots where confirmed bookings are short-staffed, per date / station / block."""
    short = coverage[coverage["short_vs_bookings"] > 0].sort_values(["date", "station", "slot"])
    gaps = []
    for (d, station), g in short.groupby(["date", "station"], sort=False):
        run = None
        for r in g.itertuples(index=False):
            block = "AM" if r.slot < 16 else "PM"
            if run and r.slot == run["slots"][-1] + sc.SLOT and block == run["block"]:
                run["slots"].append(r.slot)
                run["extra"] = max(run["extra"], int(r.short_vs_bookings))
                continue
            run = {"date": d, "station": station, "block": block, "slots": [r.slot],
                   "extra": int(r.short_vs_bookings)}
            gaps.append(run)
    for g in gaps:
        g["label"] = (f"{g['station']}, {g['date']:%a %d %b} {sc.hhmm(g['slots'][0])}-"
                      f"{sc.hhmm(g['slots'][-1] + sc.SLOT)}, {g['extra']} more {'person' if g['extra'] == 1 else 'people'}")
    return gaps


def _covers(shift, slots):
    return all(shift.on_floor(x) for x in slots)


def _week_hours(schedule, sid, d):
    mine = schedule[schedule["staff_id"] == sid]
    return mine[mine["date"].map(sc.week_of) == sc.week_of(d)]["paid_hours"].sum()


def _hours_ok(person, need):
    """Extra contract hours are only suggested within the law (48 h) and the student limit (20 h)."""
    return need <= (20 if person["contract"] == "Student" else 48)


def suggest_fixes(schedule, coverage, staff, availability, per_gap=3):
    """Ways to cover each gap against confirmed bookings, cheapest and simplest first:
    extend a shift / add a shift (no Run needed), ask someone marked unavailable, agree
    extra contract hours, and hiring when a role is short overall. Every option is
    trial-run against all the rules before it is suggested."""
    av = availability.set_index(["staff_id", "date", "block"])["available"]
    st = staff.set_index("staff_id")
    out, role_gap_hours, n = [], defaultdict(float), 0
    for g in gap_blocks(coverage):
        d, station, block, slots = g["date"], g["station"], g["block"], g["slots"]
        patterns = sorted([sh for sh in sc.SHIFTS if sh.block == block and _covers(sh, slots)],
                          key=lambda sh: (sh.paid_hours, sh.code))
        opts = []
        for sid in st.index:
            person = dict(st.loc[sid])
            if station not in sc.eligible_stations(person):
                continue
            nm, wage = person["first_name"], float(person["hourly_wage_gbp"])
            working = schedule[(schedule["staff_id"] == sid) & (schedule["date"] == d)]
            if len(working):
                row = working.iloc[0]
                old = sc.SHIFT_BY_CODE[row["shift_code"]]
                if row["station"] != station or old.block != block:
                    continue
                for sh in patterns:
                    if sh.paid_hours <= old.paid_hours or sh.start > old.start or sh.end < old.end:
                        continue
                    edit = {"type": "extend", "staff_id": sid, "dates": [d.isoformat()], "shift_code": sh.code,
                            "summary": f"**Extend {nm} ({sid})'s shift** on {d:%a %d %b} at {station} from "
                                       f"{row['start']}-{row['end']} to {sc.hhmm(sh.start)}-{sc.hhmm(sh.end)}"}
                    _, probs = apply_schedule_edit(schedule, staff, availability, edit)
                    if not probs:
                        extra = (sh.paid_hours - old.paid_hours) * wage
                        opts.append((0, extra, {"kind": "extend", "changes": [edit], "extra_cost": extra,
                                                "needs_run": False,
                                                "text": edit["summary"] + f" (+{sh.paid_hours - old.paid_hours:g} h, "
                                                        f"+£{extra:,.0f}). {nm} is already on {station} that day."}))
                        break
                continue
            for sh in patterns:
                edit = {"type": "add", "staff_id": sid, "dates": [d.isoformat()], "shift_code": sh.code,
                        "station": station,
                        "summary": f"**Add a shift for {nm} ({sid})** on {d:%a %d %b} at {station}, "
                                   f"{sc.hhmm(sh.start)}-{sc.hhmm(sh.end)}"}
                cost = sh.paid_hours * wage
                left = person["max_weekly_hours"] - _week_hours(schedule, sid, d)
                if av.get((sid, d, block), 0) == 1:
                    _, probs = apply_schedule_edit(schedule, staff, availability, edit)
                    if not probs:
                        opts.append((1, cost, {"kind": "add", "changes": [edit], "extra_cost": cost, "needs_run": False,
                                               "text": edit["summary"] + f" ({sh.paid_hours:g} h, +£{cost:,.0f}). "
                                                       f"{nm} is free and has {left:g} h left this week."}))
                        break
                    if all("over their" in p for p in probs):
                        need = _week_hours(schedule, sid, d) + sh.paid_hours
                        if _hours_ok(person, need):
                            hours = {"type": "max_hours", "staff_id": sid, "max_weekly_hours": float(need),
                                     "summary": f"**{nm} ({sid})**: max hours {person['max_weekly_hours']:g} → "
                                                f"**{need:g}** h/week (agreed extra hours)"}
                            opts.append((3, cost, {"kind": "more_hours", "changes": [hours, edit], "extra_cost": cost,
                                                   "needs_run": False,
                                                   "text": f"**Agree extra hours with {nm}** (max {person['max_weekly_hours']:g} → "
                                                           f"{need:g} h/week), then " + edit["summary"][0].lower()
                                                           + edit["summary"][1:] + f" (+£{cost:,.0f})."}))
                            break
                else:
                    trial_av = availability.copy()
                    trial_av.loc[(trial_av["staff_id"] == sid) & (trial_av["date"] == d)
                                 & (trial_av["block"] == block), "available"] = 1
                    _, probs = apply_schedule_edit(schedule, staff, trial_av, edit)
                    if not probs:
                        ask = {"type": "availability", "staff_id": sid, "available": 1, "dates": [d.isoformat()],
                               "blocks": [block],
                               "summary": f"**{nm} ({sid})**: **available** on {d:%a %d %b}, "
                                          f"{_block_text([block])} (agreed with them)"}
                        opts.append((2, cost, {"kind": "ask", "changes": [ask, edit], "extra_cost": cost,
                                               "needs_run": False,
                                               "text": f"**Ask {nm}** if they can work {d:%a %d %b} "
                                                       f"{'morning' if block == 'AM' else 'evening'} (currently marked "
                                                       f"unavailable): {station} {sc.hhmm(sh.start)}-{sc.hhmm(sh.end)} "
                                                       f"(+£{cost:,.0f})."}))
                        break
                    need = _week_hours(schedule, sid, d) + sh.paid_hours
                    if all("over their" in p for p in probs) and _hours_ok(person, need):
                        ask = {"type": "availability", "staff_id": sid, "available": 1, "dates": [d.isoformat()],
                               "blocks": [block],
                               "summary": f"**{nm} ({sid})**: **available** on {d:%a %d %b}, "
                                          f"{_block_text([block])} (agreed with them)"}
                        hours = {"type": "max_hours", "staff_id": sid, "max_weekly_hours": float(need),
                                 "summary": f"**{nm} ({sid})**: max hours {person['max_weekly_hours']:g} → "
                                            f"**{need:g}** h/week (agreed extra hours)"}
                        opts.append((4, cost, {"kind": "ask_more", "changes": [ask, hours, edit], "extra_cost": cost,
                                               "needs_run": False,
                                               "text": f"**Ask {nm}** to come in {d:%a %d %b} "
                                                       f"{'morning' if block == 'AM' else 'evening'} (currently "
                                                       f"unavailable) **and** agree extra hours this week "
                                                       f"(max {person['max_weekly_hours']:g} → {need:g} h): {station} "
                                                       f"{sc.hhmm(sh.start)}-{sc.hhmm(sh.end)} (+£{cost:,.0f})."}))
                        break
        opts.sort(key=lambda o: (o[0], o[1]))
        gap_hours = len(slots) * sc.SLOT * g["extra"]
        if not opts and gap_hours <= 1:
            n += 1
            out.append({"id": n, "gap": g["label"], "kind": "note", "changes": [], "extra_cost": 0,
                        "needs_run": False,
                        "text": f"Small gap ({gap_hours * 60:g} min), usually two breaks overlapping. Ask the "
                                f"{station} team on shift to stagger their breaks, or accept it: no one else can "
                                f"legally cover it."})
            continue
        if not opts:
            role = next(iter(sc.STATIONS[station]["roles"]))
            role_gap_hours[(role, sc.STATIONS[station]["senior_only"])] += len(slots) * sc.SLOT * g["extra"]
        for _, _, o in opts[:per_gap]:
            n += 1
            out.append(dict(o, id=n, gap=g["label"]))
        if g["extra"] > 1 and opts:
            out[-1]["text"] += f" (This gap needs {g['extra']} people, so you may need more than one option.)"
    # Hiring: roles with a gap nobody on the team can cover, or short by 4+ staff-hours overall.
    total_by_role = defaultdict(float)
    for g in gap_blocks(coverage):
        key = (next(iter(sc.STATIONS[g["station"]]["roles"])), sc.STATIONS[g["station"]]["senior_only"])
        total_by_role[key] += len(g["slots"]) * sc.SLOT * g["extra"]
    work = staff.copy()
    for (role, senior), hours in total_by_role.items():
        if hours < 4:
            continue
        same = staff[(staff["role"] == role) & ((staff["seniority"] == "Senior") if senior else True)]
        wage = float(same["hourly_wage_gbp"].median()) if len(same) else 14.0
        title = {"Chef": "Senior Chef de Partie" if senior else "Chef de Partie", "Server": "Waiter",
                 "Kitchen Porter": "Kitchen Porter", "Bartender": "Bartender", "Host": "Host",
                 "Manager": "Floor Supervisor"}.get(role, role)
        weekly = max(16, min(30, round(hours + 8)))
        hire = {"type": "hire", "staff_id": _next_id(work), "first_name": f"New{role.replace(' ', '')}",
                "job_title": title, "role": role, "seniority": "Senior" if senior else "Junior",
                "contract": "Part-time", "hourly_wage_gbp": wage, "max_weekly_hours": float(weekly),
                "dates": sorted({d.isoformat() for d in availability["date"].dropna().unique()}),
                "blocks": ["AM", "PM"]}
        work = pd.concat([work, pd.DataFrame([{"staff_id": hire["staff_id"]}])], ignore_index=True)
        hire["summary"] = (f"**Hire a part-time {title}** ({hire['staff_id']}, placeholder name {hire['first_name']}), "
                           f"£{wage:.2f}/h, up to {weekly} h/week")
        n += 1
        out.append({"id": n, "gap": f"{role} cover overall ({hours:g} staff-hours short)", "kind": "hire",
                    "changes": [hire], "extra_cost": wage * weekly, "needs_run": True,
                    "text": hire["summary"] + f" (about £{wage * weekly:,.0f}/week). The team is {hours:g} staff-hours "
                            f"short on {role.lower()} cover" + ("" if role_gap_hours.get((role, senior)) else
                                                               ", even though some gaps can be patched this week")
                            + ". After applying, click **Run** so the optimiser can use them."})
    return out


def _suggestion_for(s, people, days, suggestions):
    """Suggestions the manager is agreeing to in their own words, e.g. 'ok, go with asking Jun
    for Friday': agreement words + a person (and optionally a day) that appear in a suggestion."""
    if not suggestions or not re.search(r"\b(ok|okay|yes|go with|let'?s|do|ask|use|book|get|agree|sounds good)\b", s):
        return []
    wanted = {d.isoformat() for d in days}
    matches = []
    for x in suggestions:
        if not x["changes"]:
            continue
        who = {c.get("staff_id") for c in x["changes"]} | {c.get("other_id") for c in x["changes"]}
        when = {d for c in x["changes"] for d in (c.get("dates") or [])}
        if set(people) & who and (not wanted or wanted & when):
            matches.append(x)
    return matches


def replacement_plan(schedule, staff, availability, sid, dates, blocks=("AM", "PM"), per_shift=3):
    """Who could take each of `sid`'s shifts on these dates, best first.

    Each candidate takes the same shift (same time and station), must be allowed at that
    station and pass every rule. Ranked by: prefers that shift, then cost, then hours left.
    Shifts are planned in date order and the top pick is assumed for later shifts, so the
    same person isn't offered more hours than they can legally do."""
    want = {date.fromisoformat(d) if isinstance(d, str) else d for d in dates}
    rows = schedule[(schedule["staff_id"] == sid) & schedule["date"].isin(want)]
    if rows.empty:
        return []
    keep = [sc.SHIFT_BY_CODE[c].block in blocks for c in rows["shift_code"]]
    rows = rows[keep].sort_values(["date", "start"])
    pref = availability.set_index(["staff_id", "date", "block"])["preference"]
    st_ = staff.set_index("staff_id")
    trial, plan = schedule, []
    for r in rows.to_dict("records"):
        block = sc.SHIFT_BY_CODE[r["shift_code"]].block
        options = []
        for other in st_.index:
            if other == sid or r["station"] not in sc.eligible_stations(dict(st_.loc[other])):
                continue
            edit = {"type": "reassign", "staff_id": sid, "other_id": other, "dates": [r["date"].isoformat()]}
            new, probs = apply_schedule_edit(trial, staff, availability, edit)
            kind, extra = "direct", []
            if probs:
                # Could they do it if they agree to come in and / or to extra hours?
                if not all(("is not available" in x) or ("over their" in x) for x in probs):
                    continue
                person = dict(st_.loc[other])
                nm = person["first_name"]
                if any("is not available" in x for x in probs):
                    extra.append({"type": "availability", "staff_id": other, "available": 1,
                                  "dates": [r["date"].isoformat()], "blocks": [block],
                                  "summary": f"**{nm} ({other})**: **available** on {r['date']:%a %d %b}, "
                                             f"{_block_text([block])} (agreed with them)"})
                if any("over their" in x for x in probs):
                    need = _week_hours(trial, other, r["date"]) + r["paid_hours"]
                    if not _hours_ok(person, need):
                        continue
                    extra.append({"type": "max_hours", "staff_id": other, "max_weekly_hours": float(need),
                                  "summary": f"**{nm} ({other})**: max hours {person['max_weekly_hours']:g} → "
                                             f"**{need:g}** h/week (agreed extra hours)"})
                t_staff, t_av = apply_changes(staff, availability, extra)
                new, probs = apply_schedule_edit(trial, t_staff, t_av, edit)
                if probs:
                    continue
                kind = "+".join(("ask" if c["type"] == "availability" else "extra hours") for c in extra)
            p = int(pref.get((other, r["date"], block), 0))
            wage = float(st_.loc[other, "hourly_wage_gbp"])
            left = st_.loc[other, "max_weekly_hours"] - _week_hours(new, other, r["date"])
            options.append({"staff_id": other, "name": st_.loc[other, "first_name"],
                            "job": st_.loc[other, "job_title"], "cost": r["paid_hours"] * wage, "wage": wage,
                            "preference": p, "hours_left": left, "kind": kind, "changes": extra,
                            "_schedule": new})
        options.sort(key=lambda o: (o["kind"] != "direct", len(o["changes"]), -o["preference"], o["cost"],
                                    -o["hours_left"]))
        options = options[:per_shift]
        plan.append({"date": r["date"], "station": r["station"], "start": r["start"], "end": r["end"],
                     "block": block, "paid_hours": r["paid_hours"],
                     "label": f"{r['date']:%a %d %b}, {r['station']} {r['start']}-{r['end']}",
                     "options": [{k: v for k, v in o.items() if k != "_schedule"} for o in options]})
        if options:
            trial = options[0]["_schedule"]
    return plan


def default_pick(item):
    """Index of the first option that needs nobody's agreement, or -1 (let the optimiser decide)."""
    return next((i for i, o in enumerate(item["options"]) if o["kind"] == "direct"), -1)


def plan_pick(message, plans, staff, all_dates):
    """Changes to the dropdown picks from a typed message, e.g. 'give Saturday to Ravi',
    'Omar can do Sunday' or 'let the optimiser decide'. Returns {(plan, item): option or -1}."""
    s = message.lower()
    if re.search(r"\b(optimi[sz]er|let shiftr|leave (it|them)|run it|automatic)\b", s):
        return {(p, i): -1 for p, plan in enumerate(plans) for i, _ in enumerate(plan["items"])}
    people, _ = find_people(message, staff)
    days, _ = find_dates(message, sorted(all_dates))
    picks = {}
    for p, plan in enumerate(plans):
        for i, item in enumerate(plan["items"]):
            if days and item["date"] not in days:
                continue
            for k, o in enumerate(item["options"]):
                if o["staff_id"] in people:
                    picks[(p, i)] = k
    if not days:   # a name with no day only counts if it fits exactly one shift
        if len(picks) != 1:
            return {}
    return picks


def option_label(o):
    pref = {1: ", prefers this shift", -1: ", would rather not"}.get(o["preference"], "")
    needs = {"direct": "", "ask": " (ask first: marked unavailable)",
             "extra hours": " (needs their OK to extra hours)",
             "ask+extra hours": " (ask first: marked unavailable, and needs extra hours)"}[o["kind"]]
    left = f", {o['hours_left']:g} h left that week" if o["kind"] == "direct" else ""
    return f"{o['name']} ({o['job']}, £{o['wage']:.2f}/h, +£{o['cost']:,.0f}{left}{pref}){needs}"


def plan_text(name, plan):
    if not plan:
        return (f"{name} has no shifts on those days in the period being scheduled, so there's nothing to hand "
                f"over now. When you schedule later weeks, the optimiser won't use them.")
    lines = [f"**{name}'s shifts this period and who could take them** (best first, all rule-checked):"]
    for item in plan:
        if item["options"]:
            opts = " · ".join(f"{'**' if i == 0 else ''}{chr(65 + i)}) {option_label(o)}{'**' if i == 0 else ''}"
                              for i, o in enumerate(item["options"]))
            lines.append(f"- **{item['label']}**: {opts}")
        else:
            lines.append(f"- **{item['label']}**: nobody can take this exact shift without breaking a rule. "
                         f"Choose *let the optimiser decide*: when you click **Run** it can reshuffle other "
                         f"people's shifts to cover it.")
    lines.append("Below, each shift has a dropdown with the best option that needs nobody's agreement already "
                 "picked (or *let the optimiser decide*). Change any of them, or type e.g. *give Saturday to Ravi*, "
                 "then click **Apply**.")
    return "\n".join(lines)


def list_suggestions(suggestions):
    if not suggestions:
        return "No suggestions: every confirmed booking is covered."
    lines, last_gap = [], None
    for x in suggestions:
        if x["gap"] != last_gap:
            lines.append(f"\n**{x['gap']}**")
            last_gap = x["gap"]
        lines.append(f"{x['id']}. {x['text']}")
    return "\n".join(lines).strip()


def explain_suggestion(sug):
    kinds = {"extend": "This keeps the same person on the station a little longer, so no new shift is needed.",
             "add": "This puts someone who is free that day on an extra shift.",
             "ask": "This person is marked unavailable, so only do this if they agree to come in.",
             "more_hours": "Their contract hours are the only thing blocking this, so it needs their agreement "
                           "to work more hours this week (still within the 48-hour legal limit).",
             "ask_more": "They're marked unavailable and would also go over their contract hours, so this needs "
                         "both their agreement to come in and to work extra hours (still within the legal limits).",
             "hire": "Hiring fixes a recurring shortage rather than one shift.",
             "note": "There's no legal way to cover this from the current team, and it's too small to hire for."}
    return (f"**Suggestion {sug['id']}** for *{sug['gap']}*: {sug['text']}\n\n{kinds[sug['kind']]} "
            + ("" if not sug["changes"] else
               f"Shiftr has already checked it against availability, roles, rest, days off and weekly hours. "
               f"Say *do suggestion {sug['id']}* to apply it."))


def staff_workbook(staff, availability):
    """Excel bytes of the updated staff data, in the same layout the app reads."""
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        staff.to_excel(xw, sheet_name="Staff", index=False)
        out = availability.sort_values(["staff_id", "date", "block"]).copy()
        out.to_excel(xw, sheet_name="Availability", index=False)
    return buf.getvalue()
