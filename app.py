"""Shiftr manager dashboard (Streamlit).

The manager uploads three Excel files (staff availability, bookings, last year's
history), then clicks Run to rebuild the master schedule. It also builds once
when the page first opens. The schedule is shown on the page and can be downloaded as Excel.

The interface only collects inputs and displays results. All calculation is in
shiftr_core.py, the same engine the notebook uses.

Run on the ManSci VM (from a notebook in the same folder):
    from mansci_tools import run_app
    run_app("app.py", kind="streamlit")
"""

import base64
import io
import json
import os
import tempfile
import sys
from datetime import date, datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import declare_component

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # find shiftr_core.py next to this file
import shiftr_core as sc  # noqa: E402
import shiftr_chat as ch  # noqa: E402
import shiftr_ai as ai  # noqa: E402

# Sample Excel files live in data/, or next to app.py if the folder was flattened
# (e.g. when uploaded through GitHub's web page).
DATA = HERE / "data" if (HERE / "data").is_dir() else HERE
DEFAULT_FILES = {
    "staff": DATA / "Shiftr_Staff_Availability_Oct2026.xlsx",
    "bookings": DATA / "Shiftr_October_2026_Synthetic_Bookings.xlsx",
    "history": DATA / "Shiftr_October_2025_Historical_Demand.xlsx",
}
EXPECTED = {
    "staff": "sheets **Staff** (staff_id, first_name, job_title, role, seniority, contract, "
             "hourly_wage_gbp, max_weekly_hours, preferred_block) and **Availability** "
             "(staff_id, date, block, available, preference)",
    "bookings": "sheet **Bookings** (Booking ID, Reservation Date, Reservation Time, Meal Period, "
                "Covers (confirmed), Event Type)",
    "history": "sheet **Daily Demand** (Date, Day, Walk-in Lunch Customers, Walk-in Dinner Customers, "
               "Bar Walk-in Customers)",
}
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

st.set_page_config(page_title="Shiftr", page_icon=sc.logo_png(size=64), layout="wide")


def apply_brand_css(main, second):
    """Recolour the dashboard with the firm's two colours. Text colour is picked
    automatically (black or white) so it stays readable on each background."""
    on_main, on_second = sc.readable_text(main), sc.readable_text(second)
    st.markdown(f"""
<style>
  /* sidebar: brand background, readable text on it */
  [data-testid="stSidebar"] {{ background-color: {second}; }}
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
  [data-testid="stSidebar"] [data-testid="stExpander"] summary *,
  [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
  [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {{ color: {on_second}; }}
  /* sidebar inputs: always white with black text, whatever the brand colour */
  [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
  [data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
  [data-testid="stSidebar"] [data-baseweb="select"] > div,
  [data-testid="stSidebar"] button[kind="secondary"],
  [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
  [data-testid="stSidebar"] [data-testid="stExpander"] details {{
      background-color: #FFFFFF !important; border-color: #BFBFBF !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
  [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] *,
  [data-testid="stSidebar"] [data-baseweb="select"] > div *,
  [data-testid="stSidebar"] button[kind="secondary"] *,
  [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *,
  [data-testid="stSidebar"] [data-testid="stExpander"] details * {{ color: #000000 !important; }}
  button[kind="primary"], [data-testid="stBaseButton-primary"] {{
      background-color: {main} !important; border-color: {main} !important; color: {on_main} !important; }}
  button[kind="primary"] *, [data-testid="stBaseButton-primary"] * {{ color: {on_main} !important; }}
  [data-testid="stSidebar"] [data-testid="stColorPickerBlock"] {{
      outline: 2px solid {on_second}; outline-offset: 2px; }}
  h1, h2, h3 {{ color: {main} !important; }}
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {on_second} !important; }}
  [data-baseweb="tab-highlight"] {{ background-color: {main} !important; }}
  [data-baseweb="tab"][aria-selected="true"] p {{ color: {main} !important; }}
  [data-testid="stMetricValue"] {{ color: {main}; }}
  [data-testid="stSlider"] [role="slider"] {{ background-color: {main} !important; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data loading

# Excel upload box that sends the file over Streamlit's live connection (the same one
# buttons and sliders use). The standard st.file_uploader sends a separate upload
# request, which the ManSci proxy drops, so uploads never reached the app.
UPLOAD_COMPONENT_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  body { margin: 0; font-family: "Source Sans Pro", "Source Sans 3", sans-serif; font-size: 14px; }
  .lbl { margin: 0 0 6px; }
  .box { border: 1px dashed var(--line); border-radius: 8px; padding: 10px; display: flex;
         gap: 8px; align-items: center; flex-wrap: wrap; background: var(--bg2); }
  .box.drag { border-style: solid; border-color: var(--accent); }
  button { font: inherit; border-radius: 6px; padding: 5px 12px; cursor: pointer;
           border: 1px solid var(--line); background: var(--bg); color: inherit; }
  #clear { padding: 2px 8px; }
  .name { flex: 1; min-width: 120px; overflow-wrap: anywhere; opacity: .85; }
  .ok { color: var(--accent); opacity: 1; }
</style></head>
<body>
  <div class="lbl" id="lbl"></div>
  <div class="box" id="box">
    <button id="pick" type="button">Browse files</button>
    <span class="name" id="name">No file chosen (using the built-in sample)</span>
    <button id="clear" type="button" title="Remove file" hidden>&#x2715;</button>
    <input type="file" id="file" accept=".xlsx" hidden>
  </div>
<script>
  // Minimal Streamlit component: reads the Excel file in the browser and sends it to
  // the app over Streamlit's own connection (no separate upload request).
  const $ = id => document.getElementById(id);
  const send = (type, data) =>
    window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, data || {}), "*");
  const fit = () => send("streamlit:setFrameHeight", {height: document.body.scrollHeight + 2});

  window.addEventListener("message", e => {
    if (!e.data || e.data.type !== "streamlit:render") return;
    $("lbl").textContent = e.data.args.label || "";
    const t = e.data.theme;
    if (t) {
      const s = document.body.style;
      s.color = t.textColor;
      s.setProperty("--bg", t.backgroundColor);
      s.setProperty("--bg2", t.secondaryBackgroundColor);
      s.setProperty("--accent", t.primaryColor);
      s.setProperty("--line", t.textColor + "55");
    }
    fit();
  });

  function useFile(file) {
    if (!file) return;
    if (!/\.xlsx$/i.test(file.name)) { $("name").textContent = "Please choose an .xlsx file"; return; }
    $("name").textContent = "Reading " + file.name + "...";
    const reader = new FileReader();
    reader.onload = () => {
      send("streamlit:setComponentValue", {dataType: "json", value: {
        name: file.name, size: file.size, data: reader.result.split(",")[1], at: Date.now()}});
      $("name").textContent = "✓ " + file.name + " (" + Math.round(file.size / 1024) + " KB) sent";
      $("name").className = "name ok";
      $("clear").hidden = false;
      fit();
    };
    reader.readAsDataURL(file);
  }

  $("pick").onclick = () => $("file").click();
  $("file").onchange = () => useFile($("file").files[0]);
  $("clear").onclick = () => {
    send("streamlit:setComponentValue", {dataType: "json", value: null});
    $("name").textContent = "No file chosen (using the built-in sample)";
    $("name").className = "name"; $("clear").hidden = true; $("file").value = ""; fit();
  };
  const box = $("box");
  box.ondragover = e => { e.preventDefault(); box.classList.add("drag"); };
  box.ondragleave = () => box.classList.remove("drag");
  box.ondrop = e => { e.preventDefault(); box.classList.remove("drag"); useFile(e.dataTransfer.files[0]); };

  send("streamlit:componentReady", {apiVersion: 1});
  fit();
</script>
</body></html>
"""


def _component_folder():
    """Folder holding the upload box's index.html (next to app.py, or a temp folder)."""
    for folder in (HERE / "upload_component", Path(tempfile.gettempdir()) / "shiftr_upload_component"):
        try:
            folder.mkdir(exist_ok=True)
            page = folder / "index.html"
            if not page.exists() or page.read_text() != UPLOAD_COMPONENT_HTML:
                page.write_text(UPLOAD_COMPONENT_HTML)
            return folder
        except OSError:
            continue
    raise RuntimeError("Could not create the upload component folder")


_excel_upload = declare_component("shiftr_excel_upload", path=str(_component_folder()))


def excel_upload(label, key):
    """(file bytes, file name) once a file has been chosen, otherwise None."""
    value = _excel_upload(label=label, key=key, default=None)
    if value and value.get("data"):
        return base64.b64decode(value["data"]), value["name"]
    return None


def file_bytes(upload, key):
    """Uploaded file if there is one, otherwise the built-in synthetic sample."""
    if upload is not None:
        return upload[0], f"{upload[1]} (uploaded)"
    return DEFAULT_FILES[key].read_bytes(), f"{DEFAULT_FILES[key].name} (built-in sample)"


@st.cache_data(show_spinner=False)
def read_inputs(staff_b, bookings_b, history_b):
    """Parse the three workbooks. Raises ValueError listing every problem found.
    Returns the data plus a list of warnings (things Shiftr fixed or assumed)."""
    problems, warnings, out = [], [], {}
    for key, part, fn, b in (("staff", "staff", sc.load_staff, staff_b),
                             ("staff", "availability", sc.load_availability, staff_b),
                             ("bookings", "bookings", sc.load_bookings, bookings_b),
                             ("history", "history", sc.load_walkin_history, history_b)):
        try:
            out[part] = fn(io.BytesIO(b))
            warnings += out[part].attrs.get("warnings", [])
        except sc.DataProblems as err:
            problems += err.problems
        except Exception as err:   # noqa: BLE001  any other parsing problem is reported to the user
            problems.append(f"The {key} file could not be read ({err}). It needs {EXPECTED[key]}.")
    if not problems:
        unknown = set(out["availability"]["staff_id"]) - set(out["staff"]["staff_id"])
        if unknown:
            problems.append(f"Availability lists staff not on the Staff sheet: {', '.join(sorted(unknown))}")
        missing_days = out["history"][out["history"]["walkin_lunch"].isna()].index.tolist()
        if missing_days:
            problems.append(f"The history file has no data for: {', '.join(missing_days)}")
    if problems:
        shown = problems[:30]
        more = f"\n- ...and {len(problems) - 30} more" if len(problems) > 30 else ""
        raise ValueError(f"**Shiftr found {len(problems)} problem(s) in the uploaded data.** "
                         f"Fix them in Excel and upload again:\n" + "\n".join(
                             p if p.startswith("- ") else f"- {p}" for p in shown) + more)
    return out["staff"], out["availability"], out["bookings"], out["history"], warnings


@st.cache_data(show_spinner=False)
def build(staff_b, bookings_b, history_b, changes_json, dates, absent, multiplier, per_server, per_chef):
    """Run the whole pipeline. Cached, so it only re-runs when an input actually changes."""
    staff, availability, bookings, walkins, _ = read_inputs(staff_b, bookings_b, history_b)
    staff, availability = ch.apply_changes(staff, availability, json.loads(changes_json))
    dates = list(dates)
    av = availability[availability["date"].isin(dates)].copy()
    av.loc[av["staff_id"].isin(absent), "available"] = 0

    demand = sc.build_demand(bookings, walkins, dates, walkin_multiplier=multiplier)
    requirements = sc.build_requirements(demand, per_server, per_chef)
    schedule, info = sc.build_schedule(staff, av, requirements, time_limit=45)
    coverage = sc.coverage_report(schedule, requirements)
    checks = sc.validate_schedule(schedule, staff, av, coverage)
    extra = sc.extra_cover_needed(coverage, "short_vs_bookings")
    grid = sc.rota_grid(schedule)

    excel = io.BytesIO()
    sc.export_excel(excel, schedule, grid, demand, coverage, checks, staff, info, extra)
    return dict(demand=demand, schedule=schedule, info=info, coverage=coverage, checks=checks,
                extra=extra, grid=grid, excel=excel.getvalue(),
                built_at=datetime.now().strftime("%H:%M:%S"))


def ai_settings():
    """AI mode settings from Streamlit secrets or environment variables (never from code).
    Returns None when no key is configured, so the rule-based assistant is used."""
    def get(name):
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:   # noqa: BLE001  no secrets file at all
            pass
        return os.environ.get(name)
    key, model = get("SHIFTR_AI_API_KEY"), get("SHIFTR_AI_MODEL")
    if not key or not model:
        return None
    return {"api_key": key, "model": model, "base_url": get("SHIFTR_AI_BASE_URL")}


def period_options(dates):
    weeks = {}
    for d in dates:
        weeks.setdefault(sc.week_of(d), []).append(d)
    options = {f"Week {v[0]:%a %d %b} - {v[-1]:%a %d %b}": tuple(v) for v in weeks.values()}
    options[f"Whole month (slow: about 2 minutes)"] = tuple(dates)
    return options


# ---------------------------------------------------------------- sidebar: uploads and settings

logo_slot = st.sidebar.empty()   # logo drawn below, once the colour scheme is known
st.sidebar.caption("Manager dashboard. Prototype using synthetic data.")

st.sidebar.subheader("1. Upload this month's data")
with st.sidebar:
    up_staff = excel_upload("Staff availability (this month)", "up_staff")
    up_bookings = excel_upload("Bookings (this month)", "up_bookings")
    up_history = excel_upload("Historic demand (same month last year)", "up_history")
st.sidebar.caption("Leave a box empty to use the built-in sample file. "
                   "After uploading, click **Run** at the bottom of this panel.")
received = st.sidebar.empty()   # filled in below once the files have been read

staff_b, staff_name = file_bytes(up_staff, "staff")
bookings_b, bookings_name = file_bytes(up_bookings, "bookings")
history_b, history_name = file_bytes(up_history, "history")

st.title("Shiftr: the right people, in the right place, at the right time")

try:
    staff, availability, bookings, walkins, data_warnings = read_inputs(staff_b, bookings_b, history_b)
except ValueError as err:
    st.error(str(err))
    st.stop()
if data_warnings:
    with st.expander(f"⚠️ Shiftr fixed or assumed {len(data_warnings)} thing(s) in your files - please check"):
        st.markdown("\n".join(f"- {w}" for w in data_warnings))

# Changes typed into the Assistant are applied on top of the files.
chat_applied = st.session_state.setdefault("chat_applied", [])
staff, availability = ch.apply_changes(staff, availability, chat_applied)
# Only staff-data changes affect the optimiser; schedule edits (swaps etc.) are applied after it.
# (Data changes bundled with a schedule edit, e.g. "agree extra hours, then add a shift", are applied
# straight away with that edit, so they don't need a re-run either.)
changes_json = json.dumps([c for c in chat_applied if c["type"] not in ch.SCHEDULE_EDITS and not c.get("with_edit")])

received.caption(
    f"Files in use:  \n"
    f"• staff: {staff_name}, {len(staff)} staff  \n"
    f"• bookings: {bookings_name}, {len(bookings)} bookings  \n"
    f"• history: {history_name}, {int(walkins['days_observed'].sum())} days"
    + (f"  \n• assistant changes: {len(chat_applied)}" if chat_applied else ""))

all_dates = sorted(availability["date"].unique())
booked_dates = set(bookings["date"])
if not booked_dates & set(all_dates):
    st.warning("None of the bookings fall in the dates covered by the staff availability file. "
               "Check both files are for the same month.")

st.sidebar.subheader("2. Choose the period")
periods = period_options(all_dates)
period = st.sidebar.selectbox("Period to schedule", list(periods), index=min(1, len(periods) - 1))

st.sidebar.subheader("3. What-if (optional)")
multiplier = st.sidebar.slider("Walk-in demand vs last year", 0.5, 2.0, 1.0, 0.1,
                               help="1.0 = same as last year. 1.3 = 30% more walk-ins.")
names = dict(zip(staff["staff_id"], staff["first_name"] + " (" + staff["job_title"] + ")"))
absent = st.sidebar.multiselect("Staff off sick for the whole period", list(names), format_func=names.get)
with st.sidebar.expander("Productivity assumptions"):
    per_server = st.slider("Diners per server", 6, 20, 12)
    per_chef = st.slider("Diners per line chef", 10, 30, 20)

# ---------------------------------------------------------------- build (on Run)

params = (staff_b, bookings_b, history_b, changes_json, periods[period], tuple(sorted(absent)),
          multiplier, per_server, per_chef)
run = st.sidebar.button("Run", type="primary", width="stretch")

st.sidebar.subheader("4. Appearance")
theme_mode = st.sidebar.radio(
    "Colour scheme", ["Standard", "Custom brand colours"], key="theme_mode",
    help="Standard follows ⋮ → Settings (Light / Dark / System). "
         "Custom uses your firm's colours on the dashboard and on printed timetables.")
brand = None
if theme_mode == "Custom brand colours":
    b1, b2 = st.sidebar.columns(2)
    main_colour = b1.color_picker("Main colour", "#7B2CBF", key="brand_main")
    second_colour = b2.color_picker("Second colour", "#F4A261", key="brand_second")
    brand = (main_colour.upper(), second_colour.upper())
    apply_brand_css(*brand)

if brand:   # sidebar background is the second colour, so the 'r' uses the main colour there
    logo_html = sc.logo_svg(brand[0], brand[1], height=44,
                            text_colour=sc.readable_text(brand[1]), r_colour=brand[0])
else:
    logo_html = sc.logo_svg(height=44)
logo_slot.markdown(logo_html, unsafe_allow_html=True)

if run or "result" not in st.session_state:   # also builds once when the page first opens
    with st.spinner("Building the master schedule... (a week takes up to a minute)"):
        try:
            all_data = json.dumps([c for c in chat_applied if c["type"] not in ch.SCHEDULE_EDITS])
            st.session_state["result"] = build(*params[:3], all_data, *params[4:])
        except RuntimeError as err:
            st.error(f"No rota could be built: {err}")
            st.stop()
    st.session_state["params"] = params
    st.session_state["inputs"] = dict(staff=staff_name, n_staff=len(staff), period=period)
    st.session_state["ran_at"] = datetime.now().strftime("%H:%M:%S")
    if run:
        st.toast(f"Schedule updated at {st.session_state['ran_at']} using {staff_name}")

r = st.session_state["result"]
run_dates, run_absent = st.session_state["params"][4], st.session_state["params"][5]
period_av = availability[availability["date"].isin(run_dates)].copy()
period_av.loc[period_av["staff_id"].isin(run_absent), "available"] = 0


def with_edits(result, edits):
    """The optimised result with the assistant's schedule edits (swaps etc.) applied on top."""
    sched, notes = result["schedule"], []
    for e in edits:
        sched, problems = ch.apply_schedule_edit(sched, staff, period_av, e)
        if problems and not e.get("soft"):
            notes.append(f"Couldn't keep *{e['summary']}* after the last run: {'; '.join(problems)}")
    req = result["coverage"][["date", "slot", "time", "station", "req_min", "req_full"]]
    cov = sc.coverage_report(sched, req)
    checks = sc.validate_schedule(sched, staff, period_av, cov)
    extra = sc.extra_cover_needed(cov, "short_vs_bookings")
    grid = sc.rota_grid(sched)
    excel = io.BytesIO()
    sc.export_excel(excel, sched, grid, result["demand"], cov, checks, staff, result["info"], extra)
    return dict(result, schedule=sched, coverage=cov, checks=checks, extra=extra, grid=grid,
                excel=excel.getvalue()), notes


edits = [c for c in chat_applied if c["type"] in ch.SCHEDULE_EDITS]
if edits:
    r, edit_notes = with_edits(r, edits)
    st.caption(f"Includes {len(edits)} manual edit(s) from the Assistant (swaps / cover / removals).")
    for n in edit_notes:
        st.warning(n)
if st.session_state["params"] != params:
    st.info("Files or settings have changed since the last run. Click **Run** in the left panel "
            "to update the schedule.")

schedule, coverage, checks, extra = r["schedule"], r["coverage"], r["checks"], r["extra"]
suggestions = ch.suggest_fixes(schedule, coverage, staff, period_av) if len(extra) else []
st.session_state.setdefault("chat_context", {})["suggestions"] = suggestions


def apply_suggestion(sug):
    """Add a suggestion's changes to the applied list and note it in the chat."""
    chat_applied.extend(dict(c, from_suggestion=sug["id"], with_edit=not sug["needs_run"], soft=True)
                        for c in sug["changes"])
    st.session_state.setdefault("chat_log", []).append(
        ("assistant", f"✓ Applied suggestion {sug['id']}: {sug['text']}"
                      + (" Click **Run** in the left panel so the optimiser can use the new hire."
                         if sug["needs_run"] else " The schedule has been updated.")))

with st.expander("Data in use", expanded=False):
    st.markdown(
        f"- **Staff availability:** {st.session_state['inputs']['staff']}: "
        f"{st.session_state['inputs']['n_staff']} staff, "
        f"{all_dates[0]:%d %b} to {all_dates[-1]:%d %b %Y}\n"
        f"- **Bookings:** {bookings_name}: {len(bookings)} bookings, {bookings['covers'].sum()} covers\n"
        f"- **Historic demand:** {history_name}: {int(walkins['days_observed'].sum())} days of history")
    t1, t2, t3 = st.columns(3)
    for col, key, label in ((t1, "staff", "staff availability"), (t2, "bookings", "bookings"),
                            (t3, "history", "historic demand")):
        col.download_button(f"Template: {label}", DEFAULT_FILES[key].read_bytes(),
                            file_name=DEFAULT_FILES[key].name, mime=XLSX, key=f"tpl_{key}")
    st.caption("Download a template, edit it in Excel (keep the sheet and column names), then upload it.")

st.caption(f"Schedule for **{st.session_state['inputs']['period']}**, last run at {st.session_state['ran_at']}. Solver: {r['info']['status']}")

passed = (checks["result"] == "PASS").sum()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Labour cost", f"£{schedule['shift_cost_gbp'].sum():,.0f}")
c2.metric("Paid hours", f"{schedule['paid_hours'].sum():,.0f}")
c3.metric("Shifts", len(schedule))
c4.metric("Demand covered", f"{sc.coverage_score(coverage):.1%}")
c5.metric("Checks passed", f"{passed} / {len(checks)}")

d1, d2 = st.columns([3, 1])
if len(extra):
    n_fix = sum(1 for x in suggestions if x["changes"])
    d1.warning(f"Confirmed bookings need **{extra['staff_hours'].sum():.1f} extra staff-hours** "
               "that the current team cannot cover. "
               + (f"Shiftr has **{n_fix} suggestion(s)** to fix this: see the *Cover gaps* tab, or ask the "
                  f"Assistant *what should I do?*" if n_fix else "See the *Cover gaps* tab."))
else:
    d1.success("Every confirmed booking is fully staffed.")
d2.download_button("Download master schedule (Excel)", r["excel"], type="primary",
                   file_name="Shiftr_Master_Schedule.xlsx", mime=XLSX, width="stretch")

chat_panel = st.expander("💬 Assistant: update staff by typing a message",
                         expanded=st.session_state.get("chat_open", False))

tab_sched, tab_grid, tab_why, tab_gaps, tab_checks = st.tabs(
    ["Master schedule", "Rota grid", "Why this many staff?", "Cover gaps", "Checks"])

with tab_sched:
    f1, f2, f3 = st.columns(3)
    days = sorted(schedule["date"].unique())
    pick_day = f1.selectbox("Day", ["All"] + days,
                            format_func=lambda d: d if d == "All" else d.strftime("%a %d %b"))
    pick_area = f2.selectbox("Area", ["All"] + sorted(schedule["area"].unique()))
    pick_person = f3.selectbox("Employee", ["All"] + sorted(schedule["employee"].unique()))
    view = schedule
    if pick_day != "All":
        view = view[view["date"] == pick_day]
    if pick_area != "All":
        view = view[view["area"] == pick_area]
    if pick_person != "All":
        view = view[view["employee"] == pick_person]
    st.dataframe(
        view[["date", "day", "employee", "job_title", "area", "placement", "start", "end",
              "break", "paid_hours", "shift_cost_gbp", "preference"]],
        hide_index=True, width="stretch")

    # Print / download exactly what the filters above are showing.
    title = "Shiftr staff timetable"
    if pick_person != "All":
        job = view["job_title"].iloc[0] if len(view) else ""
        title += f": {pick_person}" + (f" ({job})" if job else "")
    filters = [st.session_state["inputs"]["period"]]
    if pick_day != "All":
        filters.append(pick_day.strftime("%A %d %B %Y"))
    if pick_area != "All":
        filters.append(pick_area)
    subtitle = " | ".join(filters)
    slug = "_".join(x for x in [
        pick_person if pick_person != "All" else "all_staff",
        pick_day.strftime("%d%b") if pick_day != "All" else "",
        pick_area.replace(" ", "") if pick_area != "All" else ""] if x)

    sheet = io.BytesIO()
    view.drop(columns=["shift_code"], errors="ignore").to_excel(sheet, index=False, sheet_name="Timetable")
    p1, p2, _ = st.columns([1, 1, 2])
    p1.download_button("Print this timetable (PDF)", sc.timetable_pdf(view, title, subtitle, colours=brand),
                       file_name=f"Shiftr_timetable_{slug}.pdf", mime="application/pdf",
                       type="primary", width="stretch", disabled=view.empty)
    p2.download_button("Download this selection (Excel)", sheet.getvalue(),
                       file_name=f"Shiftr_timetable_{slug}.xlsx", mime=XLSX,
                       width="stretch", disabled=view.empty)
    st.caption(f"Prints the {len(view)} shifts shown above. The PDF leaves out wages and costs, "
               "so it can be handed to staff. Open it and press Ctrl/⌘ + P to print.")

with tab_grid:
    st.dataframe(r["grid"], hide_index=True, width="stretch")

with tab_why:
    st.write("Pick a day and station to see how bookings and forecast walk-ins drive the staff needed, "
             "and how many people the rota puts there (breaks are taken off the floor).")
    w1, w2 = st.columns(2)
    day = w1.selectbox("Day ", days, format_func=lambda d: d.strftime("%a %d %b"))
    station = w2.selectbox("Station", list(sc.STATIONS), index=list(sc.STATIONS).index("Dining Room"))

    d = r["demand"][r["demand"]["date"] == day]
    fig = go.Figure()
    fig.add_bar(x=d["time"], y=d["booked_food"], name="Booked diners")
    fig.add_bar(x=d["time"], y=d["walkin_food"], name="Forecast walk-in diners")
    fig.add_scatter(x=d["time"], y=d["bar"], name="Bar guests", mode="lines+markers")
    fig.add_hline(y=sc.SEATS, line_dash="dash", annotation_text=f"{sc.SEATS} seats")
    if brand:
        fig.update_layout(colorway=[brand[0], brand[1], "#555555"])
    fig.update_layout(barmode="stack", height=320, margin=dict(t=30, b=10),
                      title="Guests in the building", yaxis_title="people")
    st.plotly_chart(fig, width="stretch")

    c = coverage[(coverage["date"] == day) & (coverage["station"] == station)]
    fig2 = go.Figure()
    fig2.add_scatter(x=c["time"], y=c["req_min"], name="Needed for bookings", line_shape="hv")
    fig2.add_scatter(x=c["time"], y=c["req_full"], name="Needed incl. walk-ins", line_shape="hv")
    fig2.add_bar(x=c["time"], y=c["scheduled"], name="Scheduled on the floor", opacity=0.5)
    if brand:
        fig2.update_layout(colorway=[brand[0], brand[1], "#888888"])
    fig2.update_layout(height=320, margin=dict(t=30, b=10), title=f"{station}: staff needed vs scheduled",
                       yaxis_title="staff")
    st.plotly_chart(fig2, width="stretch")
    if d["overbooked"].any():
        st.warning("On this day, bookings alone exceed the seats at some times. "
                   "Staffing is planned for at most 50 diners; the bookings need reviewing.")

with tab_gaps:
    st.write("Times when the current team cannot staff **confirmed bookings**, after every rule is applied.")
    st.dataframe(extra, hide_index=True, width="stretch")
    if suggestions:
        st.subheader("What you could do")
        st.caption("Cheapest and simplest options first. Each one has been checked against availability, roles, "
                   "rest, days off and weekly hours. Options that need a staff member's agreement say so. "
                   "You can also ask the Assistant about any of them, e.g. *tell me more about suggestion 2*.")
        last_gap = None
        for sug in suggestions:
            if sug["gap"] != last_gap:
                st.markdown(f"**{sug['gap']}**")
                last_gap = sug["gap"]
            g1, g2 = st.columns([6, 1])
            g1.markdown(f"{sug['id']}. {sug['text']}")
            if sug["changes"] and g2.button("Apply", key=f"apply_sug_{sug['id']}_{len(chat_applied)}",
                                            width="stretch"):
                apply_suggestion(sug)
                st.rerun()
    elif len(extra):
        st.info("No legal way to cover these gaps from the current team was found. Consider agency staff "
                "or changing the bookings.")

with tab_checks:
    st.write("Each rule is re-checked directly from the finished schedule, independently of the optimiser.")
    st.dataframe(checks, hide_index=True, width="stretch")


with chat_panel:
    ai_conf = ai_settings()
    if ai_conf:
        st.caption(f"🤖 **AI mode** ({ai_conf['model']}). Chat normally: ask questions about the rota or "
                   "describe what's changed. The AI proposes; Shiftr checks every change against the rules, "
                   "and **nothing changes until you click Apply**. AI replies can contain mistakes: the "
                   "**Proposed changes** list is written by Shiftr and is exactly what will happen.")
    else:
        st.caption("📝 **Basic mode** (no AI key configured). Type short notes, one per line, e.g. "
                   "*Kofi is off sick 12-14 Oct*, *Set Nina's max hours to 16*, *Hire Sam as a waiter, 25 hours, "
                   "£14 an hour*, *Swap Francesca and Pedro on 10 Oct*, *Nadia was fired today*. "
                   "**Nothing changes until you click Apply.** Type **help** for more.")
    log = st.session_state.setdefault("chat_log", [])
    for role, text in log[-12:]:
        with st.chat_message(role):
            st.markdown(text)

    with st.form("chat_form", clear_on_submit=True):
        message = st.text_area("Message", placeholder="e.g. Hugo can't work Friday evenings",
                               label_visibility="collapsed", height=90)
        sent = st.form_submit_button("Send")
    plans = st.session_state.get("chat_plans") or []
    if sent and message.strip() and plans and st.session_state.get("chat_pending"):
        picks = ch.plan_pick(message, plans, staff, all_dates)
        if picks:
            said = []
            for (p_i, i), k in picks.items():
                st.session_state[f"plan_{st.session_state.get('plan_gen', 0)}_{p_i}_{i}"] = k
                item = plans[p_i]["items"][i]
                said.append(f"**{item['label']}** → " + ("let the optimiser decide" if k == -1 else
                                                          f"**{item['options'][k]['name']}**"))
            log += [("user", message.strip()), ("assistant", "Updated the picks:\n" + "\n".join(f"- {x}" for x in said)
                                                 + "\n\nClick **Apply** when you're happy.")]
            st.rerun()
    if sent and message.strip():
        ai_reply = None
        if ai_conf:
            try:
                with st.spinner("Thinking..."):
                    ai_reply, proposed, replies, _ = ai.ask(ai_conf, log, message, staff, schedule, all_dates,
                                                            st.session_state.setdefault("chat_context", {}))
            except Exception as err:   # noqa: BLE001  network, key or format problems -> basic mode
                st.session_state["ai_error"] = f"AI mode failed ({type(err).__name__}: {err}). Used basic mode instead."
                ai_reply = None
        if ai_reply is None:
            proposed, replies = ch.interpret(message, staff, all_dates,
                                             st.session_state.setdefault("chat_context", {}))
        trial, checked = schedule, []
        # Staff-data changes in the same message (e.g. "agree extra hours") apply before shift edits are checked.
        trial_staff, trial_av = ch.apply_changes(staff, period_av,
                                                 [c for c in proposed if c["type"] not in ch.SCHEDULE_EDITS])
        for c in proposed:
            if c["type"] in ch.SCHEDULE_EDITS:
                before_cost = trial["shift_cost_gbp"].sum()
                new, done, skipped = ch.apply_schedule_edit_partly(trial, trial_staff, trial_av, c)
                if done is None:
                    replies.append(f"I can't do *{c['summary']}* because it would break a rule: "
                                   + "; ".join(skipped) + ".")
                    continue
                req = r["coverage"][["date", "slot", "time", "station", "req_min", "req_full"]]
                gap = lambda sch: sc.extra_cover_needed(sc.coverage_report(sch, req))["staff_hours"].sum()
                gap_change = gap(new) - gap(trial)
                c = dict(done, summary=done["summary"] + f" ✓ rules OK, labour cost "
                                                         f"{new['shift_cost_gbp'].sum() - before_cost:+,.0f} GBP"
                         + (f". ⚠️ Leaves **{gap_change:g} staff-hours** of booked demand uncovered"
                            if gap_change > 0 else "")
                         + (f". **Skipped** (would break a rule): {' | '.join(skipped)}" if skipped else ""))
                trial = new
            checked.append(c)
        proposed = checked
        # Someone off / leaving with shifts in this period: plan who takes each shift.
        st.session_state["plan_gen"] = st.session_state.get("plan_gen", 0) + 1   # fresh dropdown keys
        plans = []
        for c in proposed:
            if c["type"] == "availability" and c["available"] == 0:
                items = ch.replacement_plan(trial, trial_staff, trial_av, c["staff_id"], c["dates"], c["blocks"])
                nm = staff.loc[staff["staff_id"] == c["staff_id"], "first_name"].iloc[0]
                plans.append({"staff_id": c["staff_id"], "name": nm, "items": items})
                if items:
                    replies.append(ch.plan_text(f"{nm} ({c['staff_id']})", items))
                else:
                    earlier = trial[(trial["staff_id"] == c["staff_id"])
                                    & (trial["date"] < min(date.fromisoformat(d) for d in c["dates"]))]
                    when = ", ".join(d.strftime("%a %d") for d in sorted(set(earlier["date"])))
                    replies.append(f"**{nm} has no shifts left to hand over in the period you're viewing** "
                                   f"({st.session_state['inputs']['period']})"
                                   + (f": their shifts that week ({when}) are all before "
                                      f"{min(c['dates'])[8:]} {date.fromisoformat(min(c['dates'])):%b}." if when else ".")
                                   + " To see their future shifts and who can take them, pick a later week under "
                                     "**2. Choose the period**, click **Run**, then tell me again.")
        st.session_state["chat_plans"] = plans
        reply = [ai_reply] if ai_reply else []
        if proposed:
            reply.append(("**Proposed changes** (click Apply to make them):\n" if ai_reply else
                          "Here's what I understood:\n") + "\n".join(f"- {c['summary']}" for c in proposed))
        reply += replies
        log += [("user", message.strip().replace("\n", "  \n")), ("assistant", "\n\n".join(reply) or "I didn't find anything to change.")]
        st.session_state["chat_pending"] = proposed
        st.session_state["chat_open"] = True
        st.rerun()

    if st.session_state.get("ai_error"):
        st.warning(st.session_state.pop("ai_error"))
    pending = st.session_state.get("chat_pending", [])
    if pending:
        for p_i, plan in enumerate(plans):
            if any(item["options"] for item in plan["items"]):
                st.markdown(f"**Who takes {plan['name']}'s shifts?**")
            for i, item in enumerate(plan["items"]):
                if not item["options"]:
                    continue
                key = f"plan_{st.session_state.get('plan_gen', 0)}_{p_i}_{i}"
                if key not in st.session_state:
                    st.session_state[key] = ch.default_pick(item)
                st.selectbox(item["label"], [-1] + list(range(len(item["options"]))), key=key,
                             format_func=lambda k, item=item: ("Let the optimiser decide (click Run after Apply)"
                                                               if k == -1 else ch.option_label(item["options"][k])))
        a1, a2, _ = st.columns([1, 1, 3])
        if a1.button(f"Apply {len(pending)} change(s)", type="primary", width="stretch"):
            final = [dict(c, with_edit=True, soft=True) if c.get("from_suggestion") and any(
                x["type"] in ch.SCHEDULE_EDITS for x in pending if x.get("from_suggestion") == c["from_suggestion"])
                else c for c in pending]
            handed, notes = [], []
            t_staff, t_av = ch.apply_changes(staff, period_av, [c for c in final if c["type"] not in ch.SCHEDULE_EDITS])
            t_sched = schedule
            for p_i, plan in enumerate(plans):
                if not plan["items"]:
                    continue
                gen = st.session_state.get("plan_gen", 0)
                picks = [st.session_state.get(f"plan_{gen}_{p_i}_{i}", -1) if item["options"] else -1
                         for i, item in enumerate(plan["items"])]
                left = [item["label"] for item, k in zip(plan["items"], picks) if k == -1]
                if left:
                    notes.append(f"Left to the optimiser: {'; '.join(left)}. Click **Run** to cover "
                                 f"{'it' if len(left) == 1 else 'them'} (Run re-plans the week, so check your "
                                 f"picks afterwards).")
                edits, extra, ok = [], [], True
                for item, k in zip(plan["items"], picks):
                    if k == -1:
                        continue
                    o = item["options"][k]
                    extra += [dict(c, with_edit=True) for c in o["changes"]]
                    edits.append({"type": "reassign", "staff_id": plan["staff_id"], "other_id": o["staff_id"],
                                  "dates": [item["date"].isoformat()], "soft": True,
                                  "summary": f"**{item['label']}**: {plan['name']} → **{o['name']}**"})
                s2, a2_ = ch.apply_changes(t_staff, t_av, extra)
                trial_s = t_sched
                for e in edits:
                    trial_s, probs = ch.apply_schedule_edit(trial_s, s2, a2_, e)
                    if probs:
                        ok = False
                        notes.append(f"Couldn't hand over {e['summary']} ({'; '.join(probs)}), so "
                                     f"{plan['name']}'s shifts are left to the optimiser: click **Run**.")
                        break
                if not ok:
                    continue
                t_staff, t_av, t_sched = s2, a2_, trial_s
                if not left:   # every shift handed over by hand: no re-run needed for this person
                    final = [dict(c, with_edit=True) if c["type"] == "availability"
                             and c["staff_id"] == plan["staff_id"] and c["available"] == 0 else c for c in final]
                final += extra + edits
                handed += [e["summary"].replace("**", "") for e in edits]
            chat_applied += final
            ctx = st.session_state.setdefault("chat_context", {})
            ctx["recent"] = (ctx.get("recent") or []) + handed
            needs_run = any(c["type"] not in ch.SCHEDULE_EDITS and not c.get("with_edit") for c in final)
            log.append(("assistant", f"✓ Applied {len(pending)} change(s)."
                        + (" **Shifts handed over:**\n" + "\n".join(f"- {h}" for h in handed) if handed else "")
                        + ("\n\n" + " ".join(notes) if notes else "")
                        + ("\n\nClick **Run** in the left panel to rebuild the schedule with the staff changes."
                           if needs_run and not notes else "")))
            st.session_state["chat_pending"] = []
            st.session_state["chat_plans"] = []
            st.rerun()
        if a2.button("Discard", width="stretch"):
            log.append(("assistant", "Discarded. Nothing was changed."))
            st.session_state["chat_pending"] = []
            st.session_state["chat_plans"] = []
            st.rerun()

    if chat_applied:
        st.subheader(f"Changes applied so far ({len(chat_applied)})")
        st.markdown("\n".join(f"{i}. {c['summary']}" for i, c in enumerate(chat_applied, 1)))
        u1, u2, u3 = st.columns([1, 1, 2])
        if u1.button("Undo last change", width="stretch"):
            undone = chat_applied.pop()
            log.append(("assistant", f"Undid: {undone['summary']}"))
            st.rerun()
        if u2.button("Clear all changes", width="stretch"):
            chat_applied.clear()
            log.append(("assistant", "Cleared all assistant changes. The uploaded files are used as they are."))
            st.rerun()
        u3.download_button("Download updated staff file (Excel)", ch.staff_workbook(staff, availability),
                           file_name="Shiftr_Staff_Availability_updated.xlsx", mime=XLSX,
                           width="stretch",
                           help="The staff file with these changes, in the same format as the upload. "
                                "Upload it next time to keep the changes.")
