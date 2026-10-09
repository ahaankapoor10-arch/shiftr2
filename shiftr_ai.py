"""Optional AI mode for the Shiftr assistant (natural conversation, like ChatGPT / Claude / Gemini).

The language model does two things only:
  1. writes a friendly reply (answering questions from the staff and schedule data), and
  2. rewrites whatever the manager asked for into simple plain-English COMMANDS in a fixed
     style, e.g. "Nadia has left on 9 Oct" or "Swap Francesca and Pedro on 10 Oct".
Those commands are then read by Shiftr's own rule-based reader (shiftr_chat.interpret), which
checks people, dates, wages and hours; the app then trial-runs schedule edits against every
rule, and nothing changes until the manager clicks Apply. So the AI never edits data directly,
and small local models (Ollama) only have to paraphrase, which they do far more reliably than
filling in a technical form.

Uses the OpenAI-compatible chat API (the `openai` package). Works with OpenAI, other
providers' OpenAI-compatible endpoints, and Ollama on your own laptop. Settings come from
Streamlit secrets or environment variables, never from this code:
    SHIFTR_AI_API_KEY   the provider's API key (for Ollama any text, e.g. "ollama")
    SHIFTR_AI_MODEL     the model name (e.g. an Ollama model you have pulled)
    SHIFTR_AI_BASE_URL  the OpenAI-compatible URL (Ollama: http://localhost:11434/v1;
                        leave unset for OpenAI)
"""

import json
import re
from datetime import date

import shiftr_chat as ch
import shiftr_core as sc

SYSTEM = """You are the assistant inside Shiftr, a staff-scheduling app for a 50-seat Italian restaurant
with a 10-stool bar in central London. You talk to the restaurant manager: be friendly, brief and practical.

You can answer questions about the staff and this period's schedule using the DATA below.
When the manager wants something changed, you do NOT change it yourself. Instead you write simple
COMMANDS, one per change, using ONLY these patterns (staff first names exactly as in the DATA,
dates like "9 Oct"):
  <Name> is unavailable on <date>            (add "morning" or "evening" for one shift only)
  <Name> is unavailable from <date> to <date>
  <Name> can work on <date>                  (add "morning" or "evening" for one shift only)
  <Name> has left on <date>                  (fired, quit, resigned; use today's date if none given)
  Set <Name>'s max hours to <number>
  Pay <Name> £<amount> an hour
  Hire <Name> as a <job>, <full-time|part-time|student>, <number> hours, £<amount> an hour
  Swap <Name> and <Name> on <date>           (leave out "on <date>" to swap the whole period)
  Give <Name>'s shift on <date> to <Name>
  Take <Name> off <date>                     (removes the shift, nobody replaces it)
  Add <Name> to <station> on <date> <morning|evening>   (an extra shift for someone not working that day)
  Extend <Name>'s shift on <date>            (makes their shift that day a full one)
  Apply suggestion <number>                  (one of Shiftr's numbered suggestions in the DATA)
Jobs: waiter, head waiter, bartender, head bartender, host, chef de partie, commis chef, sous chef,
head chef, pizza chef, kitchen porter, manager, supervisor.

Shiftr then checks every command against its rules (availability, one shift a day, contract hours up
to 48 a week, 11 hours' rest, a day off a week, minimum wage £{nlw}) and shows the manager a preview;
nothing changes until they click Apply. When someone leaves or is off and the manager wants the work
covered, write the "has left" / "is unavailable" command and tell them to click Run: the optimiser
then gives those shifts to the cheapest available qualified staff. Don't pick replacements yourself
unless the manager names one.
When someone is off or leaves, Shiftr itself lists their shifts with ranked replacements for the manager to
pick from, so just confirm you've noted it and tell them to choose who takes each shift below your message.
When bookings can't be fully staffed, Shiftr lists numbered SUGGESTIONS in the DATA (extend a shift, add a
shift, ask someone to come in, agree extra hours, or hire). If the manager asks what to do, explain them in
plain words, recommend one (usually the cheapest that needs least from staff) and only write
"Apply suggestion <number>" once the manager agrees. If a request is unclear (who? which day?), ask a short question and
write no commands. Never invent staff: if someone isn't in the DATA, say so.

Today is {today}. The period being scheduled is: {calendar}.
Reply with ONE JSON object only: {{"reply": "<your message to the manager>", "commands": ["...", "..."]}}
"""


def _context(staff, schedule, all_dates, suggestions=None, recent=None):
    """Staff list and schedule in a compact form (one line per person) so the prompt also
    fits the smaller context windows of local models run with Ollama."""
    lines = ["STAFF and their shifts this period (id name, job, role/seniority/contract, wage, max h/week: shifts):"]
    by_person = {}
    if schedule is not None and len(schedule):
        for r in schedule.sort_values(["date", "start"]).to_dict("records"):
            brk = "" if str(r["break"]).startswith("None") else f" brk {str(r['break'])[:5]}"
            by_person.setdefault(r["staff_id"], []).append(
                f"{r['date']:%a%d} {r['placement'].replace('Dining Room - Section ', 'Dining ')} "
                f"{r['start']}-{r['end']}{brk}")
    for r in staff.itertuples():
        shifts = "; ".join(by_person.get(r.staff_id, [])) or "no shifts"
        lines.append(f"{r.staff_id} {r.first_name}, {r.job_title}, {r.role}/{r.seniority[0]}/{r.contract}, "
                     f"£{r.hourly_wage_gbp:.2f}, {r.max_weekly_hours:g}h: {shifts}")
    lines.append(f"(Days are written like Mon05 = Monday {all_dates[0]:%B} 5.)")
    if recent:
        lines.append("\nSHIFTS RECENTLY HANDED OVER (already applied):")
        lines += [f"- {x}" for x in recent[-12:]]
    if suggestions:
        lines.append("\nCOVER GAPS (confirmed bookings short-staffed) AND SHIFTR'S SUGGESTIONS:")
        lines.append(ch.list_suggestions(suggestions).replace("**", ""))
    else:
        lines.append("\nCOVER: every confirmed booking is fully staffed this period.")
    return "\n".join(lines)


# The answer format, enforced by "structured output" where the service supports it (Ollama, OpenAI).
RESPONSE_SCHEMA = {"type": "object",
                   "properties": {"reply": {"type": "string"},
                                  "commands": {"type": "array", "items": {"type": "string"}}},
                   "required": ["reply", "commands"]}

# One worked example so smaller local models copy the style.
EXAMPLE = [
    {"role": "user", "content": "Kofi phoned in sick for tomorrow and Monday. Also how many shifts has Leila got?"},
    {"role": "assistant", "content": '{"reply": "Sorry to hear Kofi is ill. I\'ve set him as unavailable; '
                                     'click Apply, then Run, and Shiftr will cover his shifts. Leila has 6 shifts '
                                     'this period.", "commands": ["Kofi is unavailable on 10 Oct", '
                                     '"Kofi is unavailable on 12 Oct"]}'},
]


def _parse_json(text):
    """The first {...} object in the model's answer (tolerates code fences or extra words)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("the AI did not return the expected JSON")
    return json.loads(m.group(0))


def ask(settings, history, message, staff, schedule, all_dates, context=None):
    """Send the conversation to the model. Returns (reply text, changes, notes, commands).
    The model's commands go through Shiftr's own reader, which builds and checks the changes.
    Raises on connection / authentication / format errors so the app can fall back."""
    from openai import OpenAI   # imported here so the app still runs without the package

    all_dates = sorted(all_dates)
    client = OpenAI(api_key=settings["api_key"], base_url=settings.get("base_url") or None,
                    timeout=float(settings.get("timeout") or 180))   # local models can be slow
    calendar = ", ".join(d.strftime("%a %-d %b") for d in all_dates)
    system = SYSTEM.format(nlw=sc.NATIONAL_LIVING_WAGE, today=date.today().strftime("%A %-d %b %Y"),
                           calendar=calendar)
    suggestions = (context or {}).get("suggestions")
    messages = [{"role": "system", "content": system + "\n\nDATA:\n"
                 + _context(staff, schedule, all_dates, suggestions, (context or {}).get("recent"))}]
    messages += EXAMPLE
    for role, text in history[-8:]:
        messages.append({"role": "user" if role == "user" else "assistant", "content": text})
    messages.append({"role": "user", "content": message})
    # Strongest format control first; fall back for services that don't support it.
    formats = [{"type": "json_schema", "json_schema": {"name": "shiftr_answer", "schema": RESPONSE_SCHEMA}},
               {"type": "json_object"}, None]
    for i, fmt in enumerate(formats):
        try:
            extra = {"response_format": fmt} if fmt else {}
            resp = client.chat.completions.create(model=settings["model"], messages=messages, **extra)
            break
        except Exception as err:   # noqa: BLE001
            text = str(err).lower()
            if i == len(formats) - 1 or not ("response_format" in text or "json" in text or "schema" in text):
                raise
    data = _parse_json(resp.choices[0].message.content)
    commands = [str(c).strip() for c in data.get("commands") or [] if str(c).strip()]
    changes, notes = combine(message, commands, staff, all_dates, context)
    return str(data.get("reply", "")).strip(), changes, notes, commands


# Rule-based replies that only make sense in basic mode (the AI's reply covers these instead).
_CHATTY = ("but not what to change", "I couldn't find a staff member", "Which day(s)?", "I can update the staff data",
           "I can see you want to hire someone", "What job will")


def combine(message, commands, staff, all_dates, context=None):
    """Decide which changes to propose in AI mode.

    1. The manager's OWN words are read first by Shiftr's rule-based reader. If that finds
       changes, those are used: they reflect exactly what the manager typed.
    2. Otherwise the AI's commands are used as suggestions, but only for people the manager
       actually mentioned (or was just talking about), so the AI cannot add changes for staff
       nobody asked about. New hires are accepted only if the name appears in the message.
    Returns (changes, notes)."""
    context = context if context is not None else {}
    own, own_notes = ch.interpret(message, staff, all_dates, context)
    keep = [n for n in own_notes if not any(k in n for k in _CHATTY)]
    if own:
        return own, keep
    keep = [n for n in keep if "nothing to change" not in n]
    if not commands:
        return [], keep
    mentioned, _ = ch.find_people(message, staff)
    allowed = set(mentioned) | set(context.get("last_people", []))
    suggested, _ = ch.interpret("\n".join(commands), staff, all_dates, dict(context))
    ok, dropped = [], 0
    agreed = re.search(r"\b(suggestion|option|recommend|cheapest|best one|go ahead|do it|do that|yes|sounds good|"
                       r"fix|cover|sort it|that one|please)\b", message.lower())
    for c in suggested:
        people = {c.get("staff_id"), c.get("other_id")} - {None}
        named_hire = c["type"] == "hire" and c["first_name"].lower() in message.lower()
        if (c.get("from_suggestion") and agreed) or named_hire or (
                c["type"] != "hire" and people and people <= allowed):
            ok.append(dict(c, summary=c["summary"] + " *(suggested by the AI from your message)*"))
        else:
            dropped += 1
    # The model sometimes echoes a value back unchanged ("max hours 25 -> 25"); never offer those.
    ok = [c for c in ok if not ((c["type"] == "max_hours" and abs(float(staff.loc[staff["staff_id"] == c["staff_id"],
          "max_weekly_hours"].iloc[0]) - c["max_weekly_hours"]) < 0.01) or (c["type"] == "wage" and abs(float(
          staff.loc[staff["staff_id"] == c["staff_id"], "hourly_wage_gbp"].iloc[0]) - c["hourly_wage_gbp"]) < 0.005))]
    if dropped:
        keep.append(f"I ignored {dropped} AI suggestion(s) about staff you didn't mention.")
    return ok, keep
