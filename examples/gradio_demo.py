"""PULSE Temporal Embeddings -- Interactive Demo"""

import gradio as gr
import numpy as np
from datetime import datetime, timedelta
import json

# ---- Theme ----

theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fff7ed", c100="#ffedd5", c200="#fed7aa", c300="#fdba74",
        c400="#fb923c", c500="#f97316", c600="#ea580c", c700="#c2410c",
        c800="#9a3412", c900="#7c2d12", c950="#431407",
    ),
    secondary_hue=gr.themes.Color(
        c50="#fafafa", c100="#f5f5f5", c200="#e5e5e5", c300="#d4d4d4",
        c400="#a3a3a3", c500="#737373", c600="#525252", c700="#404040",
        c800="#262626", c900="#171717", c950="#0a0a0a",
    ),
    neutral_hue=gr.themes.Color(
        c50="#fafafa", c100="#f4f4f5", c200="#e4e4e7", c300="#d4d4d8",
        c400="#a1a1aa", c500="#71717a", c600="#52525b", c700="#3f3f46",
        c800="#27272a", c900="#18181b", c950="#09090b",
    ),
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0a0a0a",
    body_background_fill_dark="#0a0a0a",
    body_text_color="#d4d4d4",
    body_text_color_dark="#d4d4d4",
    body_text_color_subdued="#737373",
    body_text_color_subdued_dark="#737373",
    background_fill_primary="#111111",
    background_fill_primary_dark="#111111",
    background_fill_secondary="#0a0a0a",
    background_fill_secondary_dark="#0a0a0a",
    border_color_primary="#262626",
    border_color_primary_dark="#262626",
    block_background_fill="#111111",
    block_background_fill_dark="#111111",
    block_border_color="#1f1f1f",
    block_border_color_dark="#1f1f1f",
    block_label_background_fill="#111111",
    block_label_background_fill_dark="#111111",
    block_label_text_color="#a3a3a3",
    block_label_text_color_dark="#a3a3a3",
    block_title_text_color="#d4d4d4",
    block_title_text_color_dark="#d4d4d4",
    input_background_fill="#171717",
    input_background_fill_dark="#171717",
    input_border_color="#262626",
    input_border_color_dark="#262626",
    input_placeholder_color="#525252",
    input_placeholder_color_dark="#525252",
    button_primary_background_fill="#c2410c",
    button_primary_background_fill_dark="#c2410c",
    button_primary_background_fill_hover="#ea580c",
    button_primary_background_fill_hover_dark="#ea580c",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#1f1f1f",
    button_secondary_background_fill_dark="#1f1f1f",
    button_secondary_text_color="#d4d4d4",
    button_secondary_text_color_dark="#d4d4d4",
    button_secondary_border_color="#333333",
    button_secondary_border_color_dark="#333333",
    slider_color="#ea580c",
    slider_color_dark="#ea580c",
)

custom_css = """
.gradio-container { max-width: 960px !important; }
.dark { --color-accent: #ea580c; }
h1, h2, h3 { color: #e5e5e5 !important; }
.prose h1 { font-weight: 300 !important; letter-spacing: -0.02em; }
.prose h2 { font-weight: 400 !important; color: #a3a3a3 !important; }
.prose strong { color: #fb923c !important; }
.prose code { background: #1a1a1a !important; color: #fb923c !important; border: 1px solid #262626 !important; }
.prose table { border-collapse: collapse !important; }
.prose th { background: #171717 !important; color: #a3a3a3 !important; border: 1px solid #262626 !important; padding: 8px 12px !important; }
.prose td { background: #111111 !important; color: #d4d4d4 !important; border: 1px solid #1f1f1f !important; padding: 8px 12px !important; }
.prose hr { border-color: #262626 !important; }
.prose pre { background: #0d0d0d !important; border: 1px solid #1f1f1f !important; }
.tab-nav button { color: #737373 !important; border: none !important; }
.tab-nav button.selected { color: #ea580c !important; border-bottom: 2px solid #ea580c !important; }
footer { display: none !important; }
.label-wrap { color: #737373 !important; }
"""


# ---- Inline PULSE encoder (self-contained) ----

_TWO_PI = 2.0 * np.pi
_REF = datetime(2020, 1, 1).timestamp()
_PERIODS = np.array([3600,7200,14400,21600,28800,43200,86400,172800,604800,1209600,2592000,7776000,15552000,31557600,63115200,126230400], dtype=np.float64)
_OMEGA = 2.0 * np.pi / _PERIODS
_rng = np.random.RandomState(42)
_PHI = _rng.uniform(0, 2*np.pi, size=16)

_COG = np.array([0.20,0.15,0.12,0.10,0.12,0.18,0.30,0.50,0.70,0.85,0.95,0.92,0.85,0.72,0.68,0.75,0.88,0.90,0.82,0.70,0.55,0.40,0.30,0.25], dtype=np.float32)
_NRG = np.array([0.15,0.10,0.08,0.07,0.10,0.20,0.40,0.60,0.75,0.85,0.90,0.88,0.82,0.70,0.65,0.72,0.85,0.88,0.80,0.65,0.50,0.35,0.25,0.18], dtype=np.float32)
_HOL = [(1,1),(1,15),(2,14),(5,27),(7,4),(9,2),(10,31),(11,28),(12,25),(12,31)]

W = {"log_time":1.0,"oscillators":0.5,"circadian":1.5,"calendar":0.6,"urgency":4.0,"temporal_state":2.0,"prediction_error":3.0}

def _interp(c, h):
    h0=int(h)%24; return float(c[h0]*(1-(h-int(h)))+c[(h0+1)%24]*(h-int(h)))

def _log_time(dt):
    ts=dt.timestamp(); sm=dt.hour*3600+dt.minute*60+dt.second; doy=dt.timetuple().tm_yday
    return np.array([np.log1p(max(ts-_REF,0))/25, np.log1p(sm)/np.log1p(86400), np.log1p(dt.weekday()*86400+sm)/np.log1p(604800), (dt.hour+dt.minute/60)/24, (dt.weekday()+sm/86400)/7, (doy-1+sm/86400)/365.25, np.log1p(doy)/np.log1p(366), dt.minute/60], dtype=np.float32)

def _oscillators(dt):
    a=_OMEGA*(dt.timestamp()-_REF)+_PHI; f=np.empty(32,dtype=np.float32); f[0::2]=np.sin(a); f[1::2]=np.cos(a); return f

def _circadian(dt):
    h=dt.hour+dt.minute/60; m=dt.hour*60+dt.minute
    return np.array([np.sin(_TWO_PI*h/24),np.cos(_TWO_PI*h/24),np.sin(_TWO_PI*h/12),np.cos(_TWO_PI*h/12),np.sin(_TWO_PI*m/90),np.cos(_TWO_PI*m/90),_interp(_COG,h),_interp(_NRG,h)], dtype=np.float32)

def _calendar(dt):
    dow,mo,dom=dt.weekday(),dt.month,dt.day; doy=dt.timetuple().tm_yday; woy=dt.isocalendar()[1]; h=dt.hour+dt.minute/60; iw=float(dow>=5)
    md=366.0
    for m,d in _HOL:
        try: hd=datetime(dt.year,m,d).timetuple().tm_yday; md=min(md,min(abs(doy-hd),365-abs(doy-hd)))
        except: pass
    pk=np.array([80,172,266,355]); ds=np.minimum(np.abs(doy-pk),365-np.abs(doy-pk)); sn=np.exp(-0.5*(ds/45)**2); sn=(sn/sn.sum()).astype(np.float32)
    ct=np.array([9,14.5,19.5,3.0]); td=np.minimum(np.abs(h-ct),24-np.abs(h-ct)); tp=np.exp(-0.5*(td/3)**2); tp=(tp/tp.sum()).astype(np.float32)
    dm=[31,28,31,30,31,30,31,31,30,31,30,31][mo-1]
    return np.concatenate([[np.sin(_TWO_PI*dow/7),np.cos(_TWO_PI*dow/7)],[np.sin(_TWO_PI*(mo-1)/12),np.cos(_TWO_PI*(mo-1)/12)],[np.sin(_TWO_PI*(dom-1)/31),np.cos(_TWO_PI*(dom-1)/31)],[np.sin(_TWO_PI*woy/52),np.cos(_TWO_PI*woy/52)],[np.sin(_TWO_PI*doy/366),np.cos(_TWO_PI*doy/366)],[iw,float(md<1),float(np.exp(-md/3))],sn,tp,[float(not iw and 9<=dt.hour<17),(doy-1)/365.25,float(np.exp(-min(dom-1,dm-dom)/2))]]).astype(np.float32)

def _urgency(dt, dl_str):
    if not dl_str or not dl_str.strip(): return np.zeros(8, dtype=np.float32)
    try: dl=datetime.fromisoformat(dl_str.strip())
    except: return np.zeros(8, dtype=np.float32)
    h=(dl-dt).total_seconds()/3600
    return np.array([1/(1+.05*max(h,0)),1/(1+.5*max(h,0)),1/(1+5*max(h,0)),np.sign(h)*np.log1p(abs(h))/10,1/(1+np.exp(h*2)),np.log1p(max(-h,0))/5,.1,1/(1+.5*max(h,0))], dtype=np.float32)

def _state(dt, ev, sl):
    f=np.zeros(32,dtype=np.float32); ha=max(0,min(dt.hour+dt.minute/60-(8-min(sl,8)),18))
    f[0]=min(ev/12,1); f[1]=min(sl/10,1); f[2]=min(ha/16,1); f[3]=min(1,(ev/8)*(1-sl/10)+.1)
    f[28]=min(1,ha/14)*(1-.3*min(sl/8,1)); f[29]=.3; f[30]=float(ha>.5); return f

def encode(dt, dl="", ev=0, sl=7.0):
    parts=[_log_time(dt)*W["log_time"],_oscillators(dt)*W["oscillators"],_circadian(dt)*W["circadian"],_calendar(dt)*W["calendar"],_urgency(dt,dl)*W["urgency"],_state(dt,ev,sl)*W["temporal_state"],np.zeros(16,dtype=np.float32)]
    r=np.concatenate(parts); n=np.linalg.norm(r); return (r/n if n>0 else r).astype(np.float32)


# ---- UI Functions ----

def get_phase(h):
    if 6<=h<10: return "morning ramp", "rising"
    if 10<=h<12: return "morning peak", "peak"
    if 12<=h<14: return "post-lunch dip", "dip"
    if 14<=h<17: return "afternoon peak", "peak"
    if 17<=h<20: return "evening wind-down", "falling"
    if 20<=h<23: return "night transition", "low"
    return "deep night", "minimal"

def urgency_label(score):
    if score > 0.8: return "CRITICAL"
    if score > 0.5: return "HIGH"
    if score > 0.2: return "MODERATE"
    if score > 0.01: return "LOW"
    return "NONE"

def make_bar(value, width=20, fill="█", empty="░"):
    n = int(value * width)
    return fill * n + empty * (width - n)

def make_meter(similarity):
    """Create a visual similarity meter."""
    bar_w = 30
    filled = int(similarity * bar_w)
    bar = "█" * filled + "░" * (bar_w - filled)

    if similarity > 0.9: label, desc = "NEAR IDENTICAL", "These moments feel almost the same"
    elif similarity > 0.75: label, desc = "SIMILAR", "Experientially close moments"
    elif similarity > 0.5: label, desc = "MODERATE", "Noticeably different experiences"
    elif similarity > 0.25: label, desc = "DISTANT", "Very different temporal experiences"
    else: label, desc = "OPPOSITE", "Completely different moments"

    return f"""```
 FELT SIMILARITY

 0                              1
 ├{'─'*bar_w}┤
 │{bar}│  {similarity:.3f}
 └{'─'*bar_w}┘

 {label} — {desc}
```"""


def compare_moments(ts1, dl1, ev1, sl1, ts2, dl2, ev2, sl2):
    try:
        dt1, dt2 = datetime.fromisoformat(ts1), datetime.fromisoformat(ts2)
    except Exception as e:
        return f"**Error:** {e}", ""

    emb1, emb2 = encode(dt1, dl1, int(ev1), float(sl1)), encode(dt2, dl2, int(ev2), float(sl2))
    sim = float(np.dot(emb1, emb2))

    h1, h2 = dt1.hour + dt1.minute/60, dt2.hour + dt2.minute/60
    p1, s1 = get_phase(dt1.hour)
    p2, s2 = get_phase(dt2.hour)
    c1, c2 = _interp(_COG, h1), _interp(_COG, h2)
    e1, e2 = _interp(_NRG, h1), _interp(_NRG, h2)

    # Urgency
    u1 = _urgency(dt1, dl1)
    u2 = _urgency(dt2, dl2)
    ul1, ul2 = urgency_label(float(u1[1])), urgency_label(float(u2[1]))

    meter = make_meter(sim)

    detail = f"""| | **Moment 1** | **Moment 2** |
|:--|:--|:--|
| **When** | {dt1.strftime('%a %b %d, %I:%M %p')} | {dt2.strftime('%a %b %d, %I:%M %p')} |
| **Phase** | {p1} | {p2} |
| **Cognition** | `{make_bar(c1, 12)}` {c1:.0%} | `{make_bar(c2, 12)}` {c2:.0%} |
| **Energy** | `{make_bar(e1, 12)}` {e1:.0%} | `{make_bar(e2, 12)}` {e2:.0%} |
| **Urgency** | {ul1} | {ul2} |
| **Events** | {int(ev1)} | {int(ev2)} |
| **Sleep** | {float(sl1):.1f}h | {float(sl2):.1f}h |

---

*Clock distance between these moments: {abs((dt2-dt1).total_seconds()/3600):.1f} hours.
Felt distance: {1-sim:.3f} (0 = identical experience, 2 = polar opposite).*"""

    return meter, detail


def encode_single(ts, dl, ev, sl):
    try:
        dt = datetime.fromisoformat(ts)
    except Exception as e:
        return f"**Error:** {e}"

    emb = encode(dt, dl, int(ev), float(sl))
    h = dt.hour + dt.minute/60
    phase, intensity = get_phase(dt.hour)
    cog, eng = _interp(_COG, h), _interp(_NRG, h)

    u = _urgency(dt, dl)
    ul = urgency_label(float(u[1]))

    # Layer contributions
    layers = {
        "log_time": np.linalg.norm(_log_time(dt)*W["log_time"]),
        "oscillators": np.linalg.norm(_oscillators(dt)*W["oscillators"]),
        "circadian": np.linalg.norm(_circadian(dt)*W["circadian"]),
        "calendar": np.linalg.norm(_calendar(dt)*W["calendar"]),
        "urgency": np.linalg.norm(_urgency(dt,dl)*W["urgency"]),
        "temporal state": np.linalg.norm(_state(dt,int(ev),float(sl))*W["temporal_state"]),
    }
    total = sum(layers.values()) + 1e-6
    layer_viz = "\n".join(f"  {k:16s} {make_bar(v/total, 24)} {v/total:.0%}" for k,v in layers.items())

    # Circadian clock face (simplified)
    markers = ""
    for mark_h in [0, 6, 12, 18]:
        markers += f"{'>' if abs(dt.hour - mark_h) < 3 else ' '}"

    is_wknd = "yes" if dt.weekday() >= 5 else "no"
    is_biz = "yes" if (dt.weekday() < 5 and 9 <= dt.hour < 17) else "no"
    day_name = dt.strftime('%A')

    return f"""## {dt.strftime('%A, %B %d %Y')}
### {dt.strftime('%I:%M %p')}

---

| | |
|:--|:--|
| **Circadian phase** | {phase} ({intensity}) |
| **Cognitive capacity** | `{make_bar(cog)}` {cog:.0%} |
| **Energy level** | `{make_bar(eng)}` {eng:.0%} |
| **Urgency** | {ul} |
| **Weekend** | {is_wknd} |
| **Business hours** | {is_biz} |

---

### Layer contributions
```
{layer_viz}
```

### Raw embedding (dims 0-15)
```
{np.array2string(emb[:16], precision=3, separator=', ', max_line_width=60)}
```

*128-dimensional L2-normalized vector. Full embedding available via the Python API.*"""


def run_matrix():
    """Pre-built scenario matrix."""
    scenarios = [
        ("Mon 9am\ndeadline crunch", "2026-04-13T09:00:00", "2026-04-13T12:00:00", 6, 5),
        ("Wed 10am\ndeadline crunch", "2026-04-15T10:00:00", "2026-04-15T12:00:00", 4, 6),
        ("Sat 10am\nnothing planned", "2026-04-11T10:00:00", "", 0, 9),
        ("Mon 3am\ninsomnia", "2026-04-13T03:00:00", "", 0, 2),
        ("Fri 5pm\nweek ending", "2026-04-11T17:00:00", "", 3, 7),
        ("Tue 2pm\npost-lunch", "2026-04-14T14:00:00", "", 4, 7),
    ]

    labels = [s[0] for s in scenarios]
    embs = [encode(datetime.fromisoformat(s[1]), s[2], s[3], s[4]) for s in scenarios]
    mat = np.array([[float(np.dot(a, b)) for b in embs] for a in embs])

    # Build table
    header = "| |" + "|".join(f" **{i+1}** " for i in range(len(labels))) + "|"
    sep = "|:--|" + "|".join(":--:" for _ in labels) + "|"
    rows = []
    for i, label in enumerate(labels):
        short = label.split('\n')[0]
        cells = []
        for j in range(len(labels)):
            v = mat[i, j]
            if i == j:
                cells.append("**1.00**")
            elif v > 0.85:
                cells.append(f"**{v:.2f}**")
            elif v < 0.4:
                cells.append(f"*{v:.2f}*")
            else:
                cells.append(f"{v:.2f}")
        row = f"| **{i+1}.** {short} |" + "|".join(cells) + "|"
        rows.append(row)

    legend = "\n".join(f"**{i+1}.** {s[0].replace(chr(10), ' — ')}" for i, s in enumerate(scenarios))

    table = f"""{header}
{sep}
""" + "\n".join(rows)

    return f"""### Pairwise felt similarity

{table}

---

{legend}

---

**Bold** = high similarity (>0.85) — these moments feel alike.
*Italic* = low similarity (<0.40) — these moments feel very different.

*The two deadline crunches cluster together regardless of day.
The 3am insomnia is distant from everything.
Saturday calm and Friday evening share some relaxation energy.*"""


# ---- App ----

with gr.Blocks(theme=theme, css=custom_css, title="PULSE") as demo:

    gr.Markdown("""
# PULSE
### experiential time embeddings for AI

Time is not a number. Monday morning before a deadline and Saturday afternoon
with nowhere to be are different places in temporal space — even if a clock
treats them the same. PULSE encodes that difference.

`pip install pulse-temporal`
""")

    with gr.Tab("compare"):
        gr.Markdown("##### How similar do two moments *feel*?")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**moment 1**")
                ts1 = gr.Textbox(label="timestamp", value="2026-04-13T14:00:00", container=True)
                dl1 = gr.Textbox(label="deadline", value="2026-04-13T17:00:00", placeholder="ISO timestamp or leave empty")
                with gr.Row():
                    ev1 = gr.Slider(0, 15, value=6, step=1, label="events today")
                    sl1 = gr.Slider(0, 12, value=5, step=0.5, label="sleep hours")
            with gr.Column():
                gr.Markdown("**moment 2**")
                ts2 = gr.Textbox(label="timestamp", value="2026-04-11T14:00:00")
                dl2 = gr.Textbox(label="deadline", value="", placeholder="ISO timestamp or leave empty")
                with gr.Row():
                    ev2 = gr.Slider(0, 15, value=0, step=1, label="events today")
                    sl2 = gr.Slider(0, 12, value=9, step=0.5, label="sleep hours")

        compare_btn = gr.Button("compare moments", variant="primary", size="lg")
        meter_out = gr.Markdown()
        detail_out = gr.Markdown()
        compare_btn.click(compare_moments, [ts1,dl1,ev1,sl1,ts2,dl2,ev2,sl2], [meter_out, detail_out])

        gr.Markdown("---")
        gr.Markdown("**presets**")
        with gr.Row():
            p1 = gr.Button("crunch vs chill", size="sm")
            p2 = gr.Button("crunch vs crunch", size="sm")
            p3 = gr.Button("day vs night", size="sm")
            p4 = gr.Button("weekend vs monday", size="sm")

        p1.click(lambda: ("2026-04-13T14:00:00","2026-04-13T17:00:00",6,5,"2026-04-11T14:00:00","",0,9), outputs=[ts1,dl1,ev1,sl1,ts2,dl2,ev2,sl2])
        p2.click(lambda: ("2026-04-13T14:00:00","2026-04-13T17:00:00",6,5,"2026-04-15T10:00:00","2026-04-15T12:00:00",4,6), outputs=[ts1,dl1,ev1,sl1,ts2,dl2,ev2,sl2])
        p3.click(lambda: ("2026-04-13T10:30:00","",3,8,"2026-04-13T03:00:00","",0,2), outputs=[ts1,dl1,ev1,sl1,ts2,dl2,ev2,sl2])
        p4.click(lambda: ("2026-04-12T10:00:00","",0,9,"2026-04-13T09:00:00","2026-04-13T17:00:00",5,6), outputs=[ts1,dl1,ev1,sl1,ts2,dl2,ev2,sl2])

    with gr.Tab("encode"):
        gr.Markdown("##### What does PULSE see in a single moment?")
        ts_s = gr.Textbox(label="timestamp", value="2026-04-09T14:30:00")
        dl_s = gr.Textbox(label="deadline", value="2026-04-09T17:00:00", placeholder="optional")
        with gr.Row():
            ev_s = gr.Slider(0, 15, value=5, step=1, label="events today")
            sl_s = gr.Slider(0, 12, value=7, step=0.5, label="sleep hours")
        enc_btn = gr.Button("encode", variant="primary", size="lg")
        enc_out = gr.Markdown()
        enc_btn.click(encode_single, [ts_s, dl_s, ev_s, sl_s], [enc_out])

    with gr.Tab("matrix"):
        gr.Markdown("##### Similarity across six different temporal experiences")
        mat_btn = gr.Button("generate matrix", variant="primary", size="lg")
        mat_out = gr.Markdown()
        mat_btn.click(run_matrix, outputs=[mat_out])

    gr.Markdown("""---
<center>

**PULSE** v0.1.0 — formula-based encoder, 128D embeddings, 7 signal layers

[model card](https://huggingface.co/lalopenguin/pulse-base-v1) ·
[source](https://github.com/lalomorales22/pulse-temporal)

*word2vec showed words have geometry. PULSE shows time has geometry.*

</center>
""")

demo.launch()
