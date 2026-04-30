# =============================================================================
# B2C Sales Team Workforce Optimization — Streamlit Calculator
# =============================================================================
# Run: streamlit run streamlit_app.py
# Modes: Default (4 fixed windows) or Custom Shifts (3/4/5 user-defined)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import plotly.graph_objects as go
import plotly.express as px
import math

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Workforce Optimization Calculator",
    page_icon="MFZ",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1rem; color: #6c757d; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & REFERENCE DATA
# =============================================================================

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
HOUR_LABELS = [f"{h:02d}:00" for h in range(24)]
REF_MONTHLY = 3147

# Default 4 fixed windows
DEFAULT_WINDOWS = {
    'Night':     {'start': 0,  'end': 8,  'label': '12:00 AM – 08:00 AM'},
    'Morning':   {'start': 8,  'end': 14, 'label': '08:00 AM – 02:00 PM'},
    'Afternoon': {'start': 14, 'end': 20, 'label': '02:00 PM – 08:00 PM'},
    'Evening':   {'start': 20, 'end': 24, 'label': '08:00 PM – 12:00 AM'},
}
WINDOW_ORDER = ['Night', 'Morning', 'Afternoon', 'Evening']

# MC-calibrated reference staffing (60% avail, 95% SLA at P5)
REF_MC = {
    'weekday': {'Night': 1, 'Morning': 6, 'Afternoon': 7, 'Evening': 4},
    'weekend': {'Night': 1, 'Morning': 5, 'Afternoon': 5, 'Evening': 3},
}

# Reference window volumes (opps per window per day)
REF_WINDOW_VOL = {
    'weekday': {'Night': 12.0, 'Morning': 45.2, 'Afternoon': 44.5, 'Evening': 15.0},
    'weekend': {'Night': 10.7, 'Morning': 25.6, 'Afternoon': 27.2, 'Evening': 11.7},
}

# Reference hourly arrival rates (full dataset)
REF_HOURLY_LAMBDA = {
    'weekday': {0: 2.46, 1: 1.81, 2: 1.59, 3: 1.36, 4: 1.05, 5: 0.82,
                6: 1.38, 7: 1.58, 8: 4.96, 9: 6.22, 10: 7.22, 11: 9.13,
                12: 8.76, 13: 8.94, 14: 10.22, 15: 9.05, 16: 7.88, 17: 6.33,
                18: 5.72, 19: 5.35, 20: 5.03, 21: 3.90, 22: 3.27, 23: 2.78},
    'weekend': {0: 2.15, 1: 1.62, 2: 1.43, 3: 1.48, 4: 0.75, 5: 1.12,
                6: 0.93, 7: 1.25, 8: 2.55, 9: 3.25, 10: 4.42, 11: 5.33,
                12: 4.60, 13: 5.42, 14: 6.20, 15: 4.78, 16: 4.15, 17: 4.40,
                18: 3.45, 19: 4.20, 20: 4.05, 21: 2.98, 22: 2.30, 23: 2.33},
}

REF_HOURLY_STAFF = {
    'weekday': {**{h: 1 for h in range(0, 8)}, **{h: 6 for h in range(8, 14)},
                **{h: 7 for h in range(14, 20)}, **{h: 4 for h in range(20, 24)}},
    'weekend': {**{h: 1 for h in range(0, 8)}, **{h: 5 for h in range(8, 14)},
                **{h: 5 for h in range(14, 20)}, **{h: 3 for h in range(20, 24)}},
}

# Shift presets for custom mode
SHIFT_PRESETS = {
    3: {
        'name': '3 Shifts (8-hr blocks)',
        'shifts': [
            {'name': 'Morning',  'start': 8,  'end': 16},
            {'name': 'Evening',  'start': 16, 'end': 0},
            {'name': 'Night',    'start': 0,  'end': 8},
        ]
    },
    4: {
        'name': '4 Shifts (staggered)',
        'shifts': [
            {'name': 'Shift 1',  'start': 8,  'end': 17},
            {'name': 'Shift 2',  'start': 10, 'end': 19},
            {'name': 'Shift 3',  'start': 13, 'end': 22},
            {'name': 'Shift 4',  'start': 22, 'end': 8},
        ]
    },
    5: {
        'name': '5 Shifts (overlapping)',
        'shifts': [
            {'name': 'Early',     'start': 6,  'end': 14},
            {'name': 'Day',       'start': 9,  'end': 17},
            {'name': 'Afternoon', 'start': 13, 'end': 21},
            {'name': 'Evening',   'start': 18, 'end': 2},
            {'name': 'Night',     'start': 0,  'end': 8},
        ]
    },
}

# Shift labels for default mode
SHIFT_LABELS = {
    'Night':     '12:00 AM – 08:00 AM  (8 hrs)',
    'Morning':   '08:00 AM – 02:00 PM  (6 hrs)',
    'Afternoon': '02:00 PM – 08:00 PM  (6 hrs)',
    'Evening':   '08:00 PM – 12:00 AM  (4 hrs)',
    'Day':       '08:00 AM – 08:00 PM  (12 hrs)',
}
WINDOW_LABELS = {
    'Night':     '12:00 AM – 08:00 AM',
    'Morning':   '08:00 AM – 02:00 PM',
    'Afternoon': '02:00 PM – 08:00 PM',
    'Evening':   '08:00 PM – 12:00 AM',
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_shift_hours(start, end):
    if end == start:
        return list(range(24))
    elif end > start:
        return list(range(start, end))
    else:
        return list(range(start, 24)) + list(range(0, end))


def format_hour(h):
    h = h % 24
    if h == 0:
        return "12:00 AM"
    elif h == 12:
        return "12:00 PM"
    elif h < 12:
        return f"{h}:00 AM"
    else:
        return f"{h-12}:00 PM"


# =============================================================================
# MODEL ENGINE — DEFAULT MODE (Fixed 4 Windows)
# =============================================================================

def solve_default_lp(requirements, max_days_per_week=6):
    """LP with fixed 4 windows: Night/Morning/Afternoon/Evening + Day shift."""
    time_shifts = {
        'Day': ['Morning', 'Afternoon'], 'Morning': ['Morning'],
        'Afternoon': ['Afternoon'], 'Evening': ['Evening'], 'Night': ['Night'],
    }
    day_patterns = {}
    for i in range(7):
        working = [DAYS[(i + j) % 7] for j in range(5)]
        day_patterns[f"5d_{i}"] = working
    if max_days_per_week >= 6:
        for i, off_day in enumerate(DAYS):
            day_patterns[f"6d_{i}"] = [d for d in DAYS if d != off_day]

    patterns = {}
    for dp_name, work_days in day_patterns.items():
        for ts_name, covered in time_shifts.items():
            combo = f"{dp_name}__{ts_name}"
            cov = {}
            for d in DAYS:
                for w in WINDOW_ORDER:
                    cov[(d, w)] = 1 if (d in work_days and w in covered) else 0
            patterns[combo] = {'coverage': cov, 'work_days': work_days,
                               'time_shift': ts_name, 'days_per_week': len(work_days)}

    prob = LpProblem("Default_Shift_Opt", LpMinimize)
    x = {n: LpVariable(f"x_{n}", lowBound=0, cat='Integer') for n in patterns}
    prob += lpSum(x[n] for n in patterns)
    for d in DAYS:
        for w in WINDOW_ORDER:
            prob += lpSum(x[n] * patterns[n]['coverage'][(d, w)] for n in patterns) >= requirements[(d, w)]
    prob.solve(PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        return None

    total = int(value(prob.objective))
    assigned = []
    for n in patterns:
        val = x[n].value()
        if val and val > 0:
            sname = patterns[n]['time_shift']
            assigned.append({
                'Count': int(val), 'Shift': sname,
                'Timing': SHIFT_LABELS.get(sname, ''),
                'Days/Week': patterns[n]['days_per_week'],
                'Working Days': ', '.join(d[:3] for d in patterns[n]['work_days']),
            })
    assigned.sort(key=lambda r: -r['Count'])

    coverage = {}
    for d in DAYS:
        for w in WINDOW_ORDER:
            coverage[(d, w)] = int(sum(
                x[n].value() * patterns[n]['coverage'][(d, w)]
                for n in patterns if x[n].value() and x[n].value() > 0))

    return {'total': total, 'assigned': assigned, 'coverage': coverage, 'requirements': requirements}


def compute_default_staffing(monthly_opps, availability=0.60, sla_level='95%',
                              max_days=6, max_opps_per_rep=6):
    """Default mode: monthly volume -> headcount using fixed 4 windows."""
    sla_factor = {'75%': 0.80, '80%': 0.85, '85%': 0.88, '90%': 0.90, '95%': 1.00, '99%': 1.10}.get(sla_level, 1.00)
    avail_factor = 0.60 / availability
    volume_ratio = monthly_opps / REF_MONTHLY

    reqs = {}
    window_detail = {}
    capacity_detail = {}
    for d in DAYS:
        is_we = d in ['Saturday', 'Sunday']
        day_type = 'weekend' if is_we else 'weekday'
        for w in WINDOW_ORDER:
            ref = REF_MC[day_type][w]
            queueing_req = max(1, round(ref * np.sqrt(volume_ratio) * sla_factor * avail_factor))
            scaled_vol = REF_WINDOW_VOL[day_type][w] * volume_ratio
            capacity_req = max(1, int(math.ceil(scaled_vol / max_opps_per_rep)))
            final_req = max(queueing_req, capacity_req)
            reqs[(d, w)] = final_req
            window_detail[(day_type, w)] = final_req
            capacity_detail[(day_type, w)] = {
                'queueing': queueing_req, 'capacity': capacity_req,
                'binding': 'Capacity' if capacity_req > queueing_req else 'SLA',
                'window_vol': round(scaled_vol, 1),
            }

    sol = solve_default_lp(reqs, max_days)
    return {
        'monthly_opps': monthly_opps, 'total': sol['total'] if sol else None,
        'per_window': window_detail, 'solution': sol, 'capacity_detail': capacity_detail,
        'volume_ratio': volume_ratio, 'mode': 'default',
    }


# =============================================================================
# MODEL ENGINE — CUSTOM SHIFT MODE
# =============================================================================

def solve_custom_shift_lp(shifts, monthly_opps, availability, sla_level,
                           max_opps_per_rep, max_days_per_week=6):
    """LP with user-defined shifts, hourly coverage constraints."""
    sla_factor = {'75%': 0.80, '80%': 0.85, '85%': 0.88, '90%': 0.90, '95%': 1.00, '99%': 1.10}.get(sla_level, 1.00)
    avail_factor = 0.60 / availability
    volume_ratio = monthly_opps / REF_MONTHLY

    # Hourly requirements
    hourly_req = {}
    for dt in ['weekday', 'weekend']:
        for h in range(24):
            ref = REF_HOURLY_STAFF[dt][h]
            hourly_req[(dt, h)] = max(1, round(ref * np.sqrt(volume_ratio) * sla_factor * avail_factor))

    # Shift info
    shift_info = []
    for s in shifts:
        hours = get_shift_hours(s['start'], s['end'])
        wd_opps = sum(REF_HOURLY_LAMBDA['weekday'][h] for h in hours) * volume_ratio
        we_opps = sum(REF_HOURLY_LAMBDA['weekend'][h] for h in hours) * volume_ratio
        shift_info.append({
            **s, 'hours': hours, 'duration': len(hours),
            'wd_opps': round(wd_opps, 1), 'we_opps': round(we_opps, 1),
            'wd_cap': max(1, int(math.ceil(wd_opps / max_opps_per_rep))),
            'we_cap': max(1, int(math.ceil(we_opps / max_opps_per_rep))),
        })

    # Day patterns
    day_patterns = {}
    for i in range(7):
        working = [DAYS[(i + j) % 7] for j in range(5)]
        day_patterns[f"5d_{i}"] = working
    if max_days_per_week >= 6:
        for i, off_day in enumerate(DAYS):
            day_patterns[f"6d_{i}"] = [d for d in DAYS if d != off_day]

    prob = LpProblem("Custom_Shift_Opt", LpMinimize)
    x = {}
    for dp in day_patterns:
        for si in range(len(shift_info)):
            x[(dp, si)] = LpVariable(f"x_{dp}_S{si}", lowBound=0, cat='Integer')

    prob += lpSum(x[(dp, si)] for dp in day_patterns for si in range(len(shift_info)))

    # Hourly coverage
    for d in DAYS:
        dt = 'weekend' if d in ['Saturday', 'Sunday'] else 'weekday'
        for h in range(24):
            covering = []
            for dp, work_days in day_patterns.items():
                if d in work_days:
                    for si, s in enumerate(shift_info):
                        if h in s['hours']:
                            covering.append(x[(dp, si)])
            if covering:
                prob += lpSum(covering) >= hourly_req[(dt, h)], f"Hr_{d}_{h}"

    # Capacity per shift per day
    for d in DAYS:
        is_we = d in ['Saturday', 'Sunday']
        for si, s in enumerate(shift_info):
            cap = s['we_cap'] if is_we else s['wd_cap']
            staff_vars = [x[(dp, si)] for dp, wd in day_patterns.items() if d in wd]
            if staff_vars:
                prob += lpSum(staff_vars) >= cap, f"Cap_{d}_S{si}"

    prob.solve(PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None

    total = int(value(prob.objective))
    assigned = []
    for dp, work_days in day_patterns.items():
        for si, s in enumerate(shift_info):
            val = x[(dp, si)].value()
            if val and val > 0:
                assigned.append({
                    'Count': int(val), 'Shift': s['name'],
                    'Timing': f"{format_hour(s['start'])} – {format_hour(s['end'])}",
                    'Duration': f"{s['duration']} hrs",
                    'Days/Week': len(work_days),
                    'Working Days': ', '.join(d[:3] for d in work_days),
                })
    assigned.sort(key=lambda r: (-r['Count'], r['Shift']))

    # Coverage
    coverage = {}
    for d in DAYS:
        for h in range(24):
            covered = 0
            for dp, wd in day_patterns.items():
                if d in wd:
                    for si, s in enumerate(shift_info):
                        if h in s['hours']:
                            val = x[(dp, si)].value()
                            if val:
                                covered += int(val)
            coverage[(d, h)] = covered

    shift_by_day = {}
    for d in DAYS:
        for si, s in enumerate(shift_info):
            staff = sum(int(x[(dp, si)].value() or 0) for dp, wd in day_patterns.items() if d in wd)
            shift_by_day[(d, si)] = staff

    return {
        'total': total, 'assigned': assigned, 'coverage': coverage,
        'hourly_req': hourly_req, 'shift_info': shift_info,
        'shift_by_day': shift_by_day, 'mode': 'custom',
    }


# =============================================================================
# DATA ANALYSIS
# =============================================================================

def analyze_uploaded_data(df):
    df = df.copy()
    df['Date'] = df['Opp Created'].dt.date
    df['Hour'] = df['Opp Created'].dt.hour
    df['DOW'] = df['Opp Created'].dt.day_name()
    df['Is_Weekend'] = df['DOW'].isin(['Saturday', 'Sunday'])
    df['Year_Month'] = df['Opp Created'].dt.to_period('M')

    date_info = df[['Date', 'Is_Weekend']].drop_duplicates()
    n_wd = int((~date_info['Is_Weekend']).sum())
    n_we = int(date_info['Is_Weekend'].sum())
    total_days = len(date_info)

    wd_hourly = {h: ((~df['Is_Weekend']) & (df['Hour'] == h)).sum() / max(n_wd, 1) for h in range(24)}
    we_hourly = {h: ((df['Is_Weekend']) & (df['Hour'] == h)).sum() / max(n_we, 1) for h in range(24)}

    monthly = df.groupby('Year_Month').size().reset_index(name='Opportunities')
    monthly['Year_Month'] = monthly['Year_Month'].astype(str)
    hourly_chart = [{'Hour': f"{h:02d}:00", 'Weekday': wd_hourly[h], 'Weekend': we_hourly[h]} for h in range(24)]

    hc = df.groupby(['Date', 'Hour']).size().reset_index(name='count')
    disp = hc['count'].var() / hc['count'].mean() if hc['count'].mean() > 0 else 0
    avg_daily = len(df) / max(total_days, 1)

    return {
        'total_records': len(df), 'total_days': total_days,
        'date_range': (df['Opp Created'].min(), df['Opp Created'].max()),
        'n_weekdays': n_wd, 'n_weekends': n_we, 'avg_daily': avg_daily,
        'avg_wd_daily': len(df[~df['Is_Weekend']]) / max(n_wd, 1),
        'avg_we_daily': len(df[df['Is_Weekend']]) / max(n_we, 1),
        'est_monthly': avg_daily * 30,
        'wd_hourly': wd_hourly, 'we_hourly': we_hourly,
        'monthly_breakdown': monthly,
        'hourly_chart': pd.DataFrame(hourly_chart),
        'dispersion': disp,
        'peak_wd_hour': max(wd_hourly, key=wd_hourly.get),
        'peak_we_hour': max(we_hourly, key=we_hourly.get),
    }


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    # --- Data Input ---
    st.markdown("### Data Input")
    st.markdown("---")
    input_mode = st.radio("Input Method", ["Upload File", "Manual Entry"])

    data_stats = None
    monthly_opps = 2500

    if input_mode == "Upload File":
        uploaded = st.file_uploader("Upload Opportunity Data", type=['xlsx', 'xls', 'csv'],
                                     help="Single column: 'Opp Created' (datetime)")
        if uploaded:
            try:
                raw_df = pd.read_csv(uploaded, parse_dates=['Opp Created']) if uploaded.name.endswith('.csv') \
                    else pd.read_excel(uploaded, parse_dates=['Opp Created'])
                if 'Opp Created' not in raw_df.columns:
                    st.error(f"Column 'Opp Created' not found. Available: {', '.join(raw_df.columns.tolist())}")
                else:
                    raw_df = raw_df[['Opp Created']].dropna()
                    raw_df['Opp Created'] = pd.to_datetime(raw_df['Opp Created'])
                    data_stats = analyze_uploaded_data(raw_df)
                    monthly_opps = int(round(data_stats['est_monthly']))
                    st.success(f"Loaded {data_stats['total_records']:,} records")
            except Exception as e:
                st.error(f"Error: {e}")
        monthly_opps = st.slider("Monthly Volume (auto / override)", 500, 8000,
                                  min(max(monthly_opps, 500), 8000), 100)
    else:
        monthly_opps = st.slider("Monthly Opportunity Volume", 500, 8000, 2500, 100)

    # --- Model Parameters ---
    st.markdown("---")
    st.markdown("### Model Parameters")

    availability = st.slider("Rep Availability", 0.40, 0.85, 0.60, 0.05)
    st.caption(f"Selected: {availability:.0%}")

    sla_level = st.selectbox("SLA Target", ['75%', '80%', '85%', '90%', '95%', '99%'], index=4)

    max_opps_per_rep = st.slider("Max Opportunities / Rep / Day", 3, 10, 6, 1)

    max_days = st.radio("Max Days / Week", [5, 6], index=1,
                         format_func=lambda x: f"{x}-day work week")

    # --- Shift Mode ---
    st.markdown("---")
    st.markdown("### Shift Structure")

    shift_mode = st.radio("Shift Mode", ["Default (4 Windows)", "Custom Shifts"],
                           help="Default uses the 4 fixed operational windows. "
                                "Custom allows 3/4/5 user-defined shifts with overlap.")

    custom_shifts = None
    if shift_mode == "Custom Shifts":
        num_shifts = st.selectbox("Number of Shifts", [3, 4, 5], index=1)
        use_preset = st.checkbox("Use preset", value=True)

        if use_preset:
            preset = SHIFT_PRESETS[num_shifts]
            st.caption(f"{preset['name']}")
            for s in preset['shifts']:
                dur = len(get_shift_hours(s['start'], s['end']))
                st.caption(f"  {s['name']}: {format_hour(s['start'])} – {format_hour(s['end'])} ({dur}h)")
            custom_shifts = preset['shifts']
        else:
            custom_shifts = []
            for i in range(num_shifts):
                st.markdown(f"**Shift {i+1}**")
                c1, c2, c3 = st.columns(3)
                name = c1.text_input("Name", value=f"Shift {i+1}", key=f"sn_{i}",
                                      label_visibility="collapsed")
                start = c2.selectbox("Start", range(24), index=8,
                                      format_func=format_hour, key=f"ss_{i}")
                end = c3.selectbox("End", range(24), index=16,
                                    format_func=format_hour, key=f"se_{i}")
                custom_shifts.append({'name': name, 'start': start, 'end': end})
                dur = len(get_shift_hours(start, end))
                st.caption(f"  {format_hour(start)} – {format_hour(end)} ({dur}h)")

    st.markdown("---")
    st.markdown("### Detailed Assessment")
    show_assessment = st.button("View Assessment", use_container_width=True,
                                 type="primary")


# =============================================================================
# DETAILED ASSESSMENT DIALOG
# =============================================================================

@st.dialog("Detailed Assessment — Will This Model Deliver 95% SLA?", width="large")
def show_assessment_dialog():
    st.markdown("""
### The Short Answer

**The staffing numbers are correct — but they alone will NOT achieve the SLA target.**

The model gives you the right number of people. However, it assumes opportunities are 
assigned to reps **immediately** when they enter the pool. Your current reality is a 
**6.6-hour median assignment delay**. That gap alone breaks the 30-minute SLA before 
any rep even sees the opportunity.

---

### The Chain — And Where It Breaks

**What the model guarantees:**

Once a rep receives an opportunity, the staffing level ensures enough capacity for 95% 
of contacts to happen within the SLA window. This was validated via Monte Carlo simulation 
across 10,000 simulated days, accounting for bursty arrivals, bimodal call times 
(2 min no-answer vs 10-15 min productive call), and stochastic rep availability.

**What the model cannot fix:**

The **6.6-hour median time** from "Opportunity Created" to "Assignment to Rep." 
The descriptive statistics showed that reps respond in a **median of 4 minutes** 
after assignment. The reps are fast. The assignment mechanism is the entire problem.

---

### The Real Formula for Success

**Correct staffing + Assignment workflow fix = 95% SLA achievable**

Without the assignment fix, you could have 50 reps and still fail the SLA, because 
opportunities sit in the unassigned pool for hours regardless of how many reps are waiting.

---

### What the Assignment Fix Looks Like

**Option 1 — Auto-Assignment (Recommended):** Configure Zoho CRM round-robin. 
Opportunity is created → immediately assigned to the next available rep in rotation. 
Pool time drops from hours to seconds.

**Option 2 — Queue-Pull Model:** Reps see a live queue and pull the next opportunity 
themselves. Pool time equals the time until a rep finishes their current task.

Either approach eliminates the 6.6-hour bottleneck. With that fix in place and the 
recommended staffing on your selected shift structure, the model's 95% SLA prediction holds.

---

### Three Priority Actions

**1. Fix assignment workflow** — auto-assign or queue-pull in Zoho CRM. 
Expected impact: SLA from 15.6% to 90%+ with NO additional hires.

**2. Separate post-sale support from sales reps** — currently consuming ~30% of rep time. 
A dedicated support function pushes availability from 60% toward 70-80%, saving 2-3 FTEs.

**3. Implement recommended shift coverage** — use the model's shift roster to ensure 
peak hours (2 PM–8 PM) have adequate concurrent coverage, including weekends.

---

### Key Diagnostic Numbers

| Metric | Current State | With Fix |
|--------|--------------|----------|
| Pool Time (median) | 6.6 hours | < 1 minute |
| Contact Time (median) | 4 minutes | 4 minutes |
| 30-min SLA Adherence | 15.6% | 90-95% (projected) |
| Root Cause | Assignment delay | Eliminated |
    """)

    if st.button("Close", use_container_width=True):
        st.rerun()

if show_assessment:
    show_assessment_dialog()


# =============================================================================
# COMPUTE
# =============================================================================

is_custom = (shift_mode == "Custom Shifts") and custom_shifts is not None

if is_custom:
    result = solve_custom_shift_lp(custom_shifts, monthly_opps, availability, sla_level,
                                    max_opps_per_rep, max_days)
else:
    result = compute_default_staffing(monthly_opps, availability, sla_level, max_days, max_opps_per_rep)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown('<p class="main-header">B2C Sales Team Workforce Optimization</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Meydan Free Zone  |  Staffing Calculator</p>', unsafe_allow_html=True)

# --- Top Metrics ---
m1, m2, m3, m4, m5 = st.columns(5)
total_display = result['total'] if result else "N/A"
m1.metric("Employees Required", total_display)
m2.metric("Monthly Volume", f"{monthly_opps:,}")
m3.metric("Availability", f"{availability:.0%}")
m4.metric("SLA Target", sla_level)
m5.metric("Max Opps/Rep", max_opps_per_rep)

# --- Binding Constraint Warning ---
if not is_custom and result and result.get('capacity_detail'):
    cd = result['capacity_detail']
    all_capacity = all(v.get('binding') == 'Capacity' for v in cd.values())
    capacity_count = sum(1 for v in cd.values() if v.get('binding') == 'Capacity')
    total_count = len(cd)
    if all_capacity:
        st.warning(
            f"**All {total_count} windows are capacity-bound** (Max Opps/Rep = {max_opps_per_rep}). "
            f"The SLA target and availability settings have no impact on the result because the "
            f"workload cap is the tighter constraint everywhere. "
            f"Increase Max Opps/Rep to 8+ to see SLA and availability drive the staffing numbers.",
            icon="⚠️"
        )
    elif capacity_count > 0:
        sla_count = total_count - capacity_count
        st.info(
            f"**{capacity_count} of {total_count} windows** are capacity-bound (Max Opps/Rep = {max_opps_per_rep}), "
            f"**{sla_count}** are SLA-bound. Both constraints are active.",
            icon="ℹ️"
        )

st.markdown("---")

# --- Build Tabs ---
tab_names = []
if data_stats:
    tab_names.append("Data Analysis")
tab_names += ["Shift Roster", "Coverage", "Staffing Curve", "Scenarios"]
tabs = st.tabs(tab_names)
tab_idx = 0


# ─── TAB: Data Analysis ───
if data_stats:
    with tabs[tab_idx]:
        st.subheader("Uploaded Data Analysis")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Records", f"{data_stats['total_records']:,}")
        d2.metric("Days", data_stats['total_days'])
        d3.metric("Avg Daily", f"{data_stats['avg_daily']:.1f}")
        d4.metric("Est. Monthly", f"{data_stats['est_monthly']:,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Range:** {data_stats['date_range'][0].strftime('%Y-%m-%d')} to "
                        f"{data_stats['date_range'][1].strftime('%Y-%m-%d')}")
            st.markdown(f"**Weekdays:** {data_stats['n_weekdays']}d (avg {data_stats['avg_wd_daily']:.1f}/d)")
            st.markdown(f"**Weekends:** {data_stats['n_weekends']}d (avg {data_stats['avg_we_daily']:.1f}/d)")
        with c2:
            st.markdown(f"**Dispersion:** {data_stats['dispersion']:.2f} "
                        f"({'Bursty' if data_stats['dispersion'] > 1.5 else 'Poisson-like'})")
            st.markdown(f"**WD Peak:** {data_stats['peak_wd_hour']:02d}:00 "
                        f"({data_stats['wd_hourly'][data_stats['peak_wd_hour']]:.1f}/hr)")
            st.markdown(f"**WE Peak:** {data_stats['peak_we_hour']:02d}:00 "
                        f"({data_stats['we_hourly'][data_stats['peak_we_hour']]:.1f}/hr)")

        hc = data_stats['hourly_chart']
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(x=hc['Hour'], y=hc['Weekday'], name='Weekday', marker_color='#2c3e50', opacity=0.85))
        fig_h.add_trace(go.Bar(x=hc['Hour'], y=hc['Weekend'], name='Weekend', marker_color='#e74c3c', opacity=0.70))
        fig_h.update_layout(barmode='group', height=350, margin=dict(l=40, r=20, t=20, b=40),
                            xaxis_title="Hour", yaxis_title="Avg Arrivals/Hr",
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(fig_h, use_container_width=True)
    tab_idx += 1


# ─── TAB: Shift Roster ───
with tabs[tab_idx]:
    st.subheader("Recommended Shift Assignments")

    if result and result.get('total'):
        # Get assignments (different structure for default vs custom)
        if is_custom:
            assigned_data = result.get('assigned', [])
        else:
            assigned_data = result.get('solution', {}).get('assigned', [])

        roster_df = pd.DataFrame(assigned_data)
        roster_df.index = range(1, len(roster_df) + 1)
        roster_df.index.name = '#'
        st.dataframe(roster_df, use_container_width=True)

        if not is_custom:
            # DEFAULT MODE: show per-window table with constraint breakdown
            st.subheader("Per-Window Staffing")
            pw = result['per_window']
            cd = result.get('capacity_detail', {})
            bd_rows = []
            for dt in ['weekday', 'weekend']:
                for w in WINDOW_ORDER:
                    detail = cd.get((dt, w), {})
                    bd_rows.append({
                        'Day Type': dt.capitalize(),
                        'Window': f"{w} ({WINDOW_LABELS[w]})",
                        'Avg Opps': detail.get('window_vol', 0),
                        'SLA Req': detail.get('queueing', 0),
                        'Capacity Req': detail.get('capacity', 0),
                        'Final': pw.get((dt, w), 0),
                        'Binding': detail.get('binding', ''),
                    })
            st.dataframe(pd.DataFrame(bd_rows), use_container_width=True, hide_index=True)

            # Coverage matrix
            st.subheader("Weekly Coverage")
            cov = result['solution']['coverage']
            req = result['solution']['requirements']
            c_rows = []
            for d in DAYS:
                row = {'Day': d + (' *' if d in ['Saturday', 'Sunday'] else '')}
                for w in WINDOW_ORDER:
                    c_val, r_val = cov[(d, w)], req[(d, w)]
                    delta = c_val - r_val
                    row[f"{w} ({WINDOW_LABELS[w]})"] = f"{c_val}/{r_val}" + (f" +{delta}" if delta > 0 else "")
                c_rows.append(row)
            st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)
            st.caption("Format: Scheduled/Required. * = Weekend")

        else:
            # CUSTOM MODE: shift-by-day table
            st.subheader("Staffing by Shift and Day")
            sbd = result['shift_by_day']
            si_list = result['shift_info']
            sbd_rows = []
            for d in DAYS:
                row = {'Day': d + (' *' if d in ['Saturday', 'Sunday'] else '')}
                for si, s in enumerate(si_list):
                    col = f"{s['name']} ({format_hour(s['start'])}-{format_hour(s['end'])})"
                    row[col] = sbd.get((d, si), 0)
                row['Total'] = sum(sbd.get((d, si), 0) for si in range(len(si_list)))
                sbd_rows.append(row)
            st.dataframe(pd.DataFrame(sbd_rows), use_container_width=True, hide_index=True)
            st.caption("* = Weekend")

            # Shift info
            st.subheader("Shift Details")
            si_rows = []
            for s in si_list:
                si_rows.append({
                    'Shift': s['name'],
                    'Timing': f"{format_hour(s['start'])} – {format_hour(s['end'])}",
                    'Duration': f"{s['duration']} hrs",
                    'WD Opps': s['wd_opps'], 'WE Opps': s['we_opps'],
                    'WD Cap Req': s['wd_cap'], 'WE Cap Req': s['we_cap'],
                })
            st.dataframe(pd.DataFrame(si_rows), use_container_width=True, hide_index=True)
    else:
        st.error("No feasible solution. Try adjusting shifts or relaxing constraints.")
tab_idx += 1


# ─── TAB: Coverage ───
with tabs[tab_idx]:

    # --- Demand vs Coverage Overlay Chart (both modes) ---
    st.subheader("24-Hour Demand vs Agent Coverage")

    day_toggle = st.radio("Day Type", ["Weekday", "Weekend"], horizontal=True, key="demand_dt")
    dt_key = 'weekday' if day_toggle == "Weekday" else 'weekend'
    volume_ratio = monthly_opps / REF_MONTHLY

    hour_labels_12h = [format_hour(h) for h in range(24)]

    # Hourly opportunity arrivals (scaled to current volume)
    opps_by_hour = [REF_HOURLY_LAMBDA[dt_key][h] * volume_ratio for h in range(24)]

    # Hourly agent requirement (scaled)
    sla_factor = {'75%': 0.80, '80%': 0.85, '85%': 0.88, '90%': 0.90, '95%': 1.00, '99%': 1.10}.get(sla_level, 1.00)
    avail_factor = 0.60 / availability
    agents_by_hour = [max(1, round(REF_HOURLY_STAFF[dt_key][h] * np.sqrt(volume_ratio)
                     * sla_factor * avail_factor)) for h in range(24)]

    # Scheduled coverage (from LP solution, pick a representative day)
    if is_custom and result and result.get('coverage'):
        rep_day = 'Monday' if day_toggle == "Weekday" else 'Saturday'
        scheduled_by_hour = [result['coverage'].get((rep_day, h), 0) for h in range(24)]
    elif not is_custom and result and result.get('solution'):
        # Map window coverage to hourly for default mode
        rep_day = 'Monday' if day_toggle == "Weekday" else 'Saturday'
        cov_default = result['solution']['coverage']
        scheduled_by_hour = []
        for h in range(24):
            if h < 8:
                w = 'Night'
            elif h < 14:
                w = 'Morning'
            elif h < 20:
                w = 'Afternoon'
            else:
                w = 'Evening'
            scheduled_by_hour.append(cov_default.get((rep_day, w), 0))
    else:
        rep_day = 'Monday' if day_toggle == "Weekday" else 'Saturday'
        scheduled_by_hour = [0] * 24

    # Build dual-axis chart
    fig_demand = go.Figure()

    # Bars: opportunity arrivals
    fig_demand.add_trace(go.Bar(
        x=hour_labels_12h, y=opps_by_hour,
        name='Avg Opportunities', marker_color='#d4e6f1', opacity=0.8,
        yaxis='y',
    ))

    # Line: required agents
    fig_demand.add_trace(go.Scatter(
        x=hour_labels_12h, y=agents_by_hour,
        name='Required Agents', mode='lines+markers',
        line=dict(color='#e74c3c', width=2.5, dash='dash'),
        marker=dict(size=6), yaxis='y2',
    ))

    # Line: scheduled agents
    fig_demand.add_trace(go.Scatter(
        x=hour_labels_12h, y=scheduled_by_hour,
        name='Scheduled Agents', mode='lines+markers',
        line=dict(color='#27ae60', width=2.5),
        marker=dict(size=6), yaxis='y2',
    ))

    fig_demand.update_layout(
        height=450,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        xaxis=dict(title='Hour of Day', tickangle=-45),
        yaxis=dict(title='Opportunities / Hour', side='left', showgrid=False),
        yaxis2=dict(title='Agents', side='right', overlaying='y', showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)'),
        barmode='group',
        hovermode='x unified',
    )

    st.plotly_chart(fig_demand, use_container_width=True)
    st.caption(f"Bars = avg opportunities per hour at {monthly_opps:,}/mo. "
               f"Red dashed = minimum agents required. "
               f"Green solid = agents scheduled by LP ({rep_day}).")

    st.markdown("---")

    # --- Heatmaps (existing) ---
    if is_custom and result:
        st.subheader("Hourly Coverage Heatmap")
        view = st.radio("View", ["Scheduled", "Required", "Surplus"], horizontal=True, key="hm_v")
        cov = result['coverage']
        req = result['hourly_req']

        hm_vals = []
        for d in DAYS:
            dt = 'weekend' if d in ['Saturday', 'Sunday'] else 'weekday'
            row = []
            for h in range(24):
                if view == "Scheduled":
                    row.append(cov.get((d, h), 0))
                elif view == "Required":
                    row.append(req[(dt, h)])
                else:
                    row.append(cov.get((d, h), 0) - req[(dt, h)])
            hm_vals.append(row)

        cs = 'RdYlGn' if view == "Surplus" else 'Blues'
        fig_hm = go.Figure(data=go.Heatmap(
            z=hm_vals, x=HOUR_LABELS, y=[d[:3] for d in DAYS],
            text=[[str(v) for v in row] for row in hm_vals],
            texttemplate="%{text}", colorscale=cs, showscale=True, colorbar_title="Staff"))
        for s in result['shift_info']:
            fig_hm.add_vline(x=s['start'] - 0.5, line_dash="dot", line_color="rgba(0,0,0,0.3)",
                             annotation_text=s['name'], annotation_position="top", annotation_font_size=10)
        fig_hm.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=30),
                             xaxis_title="Hour", yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_hm, use_container_width=True)

    elif not is_custom and result and result.get('solution'):
        st.subheader("Coverage Heatmap — Default Windows")
        cov = result['solution']['coverage']
        req = result['solution']['requirements']
        hm_vals = [[cov.get((d, w), 0) for w in WINDOW_ORDER] for d in DAYS]
        x_labels = [f"{w}\n{WINDOW_LABELS[w]}" for w in WINDOW_ORDER]
        fig_hm = go.Figure(data=go.Heatmap(
            z=hm_vals, x=x_labels, y=[d[:3] for d in DAYS],
            text=[[str(v) for v in row] for row in hm_vals],
            texttemplate="%{text}", colorscale='Blues', showscale=True, colorbar_title="Reps"))
        fig_hm.update_layout(height=350, margin=dict(l=40, r=40, t=20, b=30),
                             xaxis_title="Window", yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No results to display.")
tab_idx += 1


# ─── TAB: Staffing Curve ───
with tabs[tab_idx]:
    st.subheader("Headcount vs Monthly Volume")
    vols = list(range(500, 8001, 500))
    cr = []
    for v in vols:
        for av in [0.50, 0.60, 0.70]:
            if is_custom:
                r = solve_custom_shift_lp(custom_shifts, v, av, sla_level, max_opps_per_rep, max_days)
            else:
                r = compute_default_staffing(v, av, sla_level, max_days, max_opps_per_rep)
            if r and r.get('total'):
                cr.append({'Monthly Volume': v, 'Availability': f"{av:.0%}", 'Headcount': r['total']})
    if cr:
        c_df = pd.DataFrame(cr)
        fig_c = px.line(c_df, x='Monthly Volume', y='Headcount', color='Availability',
                        color_discrete_map={'50%': '#e74c3c', '60%': '#2c3e50', '70%': '#27ae60'})
        fig_c.add_vline(x=monthly_opps, line_dash="dash", line_color="gray",
                        annotation_text=f"Selected: {monthly_opps:,}")
        fig_c.add_hline(y=17, line_dash="dot", line_color="#e74c3c",
                        annotation_text="Current Team (17)")
        fig_c.update_layout(height=450, margin=dict(l=40, r=40, t=30, b=40),
                            xaxis_title="Monthly Opportunities", yaxis_title="Employees Required",
                            legend_title="Availability", hovermode='x unified')
        st.plotly_chart(fig_c, use_container_width=True)
tab_idx += 1


# ─── TAB: Scenario Comparison ───
with tabs[tab_idx]:
    st.subheader("Multi-Scenario Comparison")
    sv = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000]
    s_rows = []
    for v in sv:
        row = {'Volume': f"{v:,}"}
        for av in [0.50, 0.60, 0.70]:
            for sla in ['75%', '80%', '85%', '90%', '95%', '99%']:
                if is_custom:
                    r = solve_custom_shift_lp(custom_shifts, v, av, sla, max_opps_per_rep, max_days)
                else:
                    r = compute_default_staffing(v, av, sla, max_days, max_opps_per_rep)
                row[f"{av:.0%}/{sla}"] = r['total'] if r and r.get('total') else '-'
        s_rows.append(row)
    st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

    mode_label = f"Custom ({len(custom_shifts)} shifts)" if is_custom else "Default (4 windows)"
    st.info(f"**{monthly_opps:,}** opps/mo | **{availability:.0%}** avail | **{sla_level}** SLA | "
            f"**{max_opps_per_rep}** max/rep | **{max_days}d** week | **{mode_label}** | "
            f"**{total_display} employees**")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

with st.expander("Model Methodology"):
    st.markdown("""
**Default Mode:** Uses 4 fixed operational windows (Night/Morning/Afternoon/Evening) with MC-calibrated 
staffing, capacity constraints, and LP shift optimization. Shift types include Day (covers Morning+Afternoon), 
individual windows, and Night.

**Custom Mode:** User-defined shifts (3/4/5) with start/end times. Supports overlapping shifts. 
LP ensures hourly coverage meets SLA requirements AND per-shift capacity constraints (max opps/rep) 
are satisfied. Overlapping shifts contribute jointly to hourly coverage.

**Calibration:** Monte Carlo simulation (10,000 days) with empirical arrivals, bimodal service times, 
and stochastic availability. Square-root staffing law for volume scaling.
    """)

with st.expander("Assumptions"):
    st.markdown(f"""
- **Service Time:** ~9 min blended (45% pickup x 12.5 min call + 55% x 2 min no-answer)
- **SLA:** 30 min from pool entry to first contact | Target: **{sla_level}**
- **Availability:** **{availability:.0%}** — post-sale support, admin, breaks
- **Max Opps/Rep:** **{max_opps_per_rep}** per day — hard workload cap (not affected by availability)
    """)

with st.expander("Input File Format"):
    st.markdown("Upload Excel/CSV with one column: **Opp Created** (datetime).")
