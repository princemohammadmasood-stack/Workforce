# =============================================================================
# B2C Sales Team Workforce Optimization — Streamlit Calculator
# =============================================================================
# Run: streamlit run streamlit_app.py
# Features: Custom shift definitions, file upload, capacity constraints
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

# MC-calibrated reference: hourly staffing needs at reference volume (3147/mo)
# Derived from Monte Carlo Layer 2 (60% availability, 95% SLA at P5)
REF_MONTHLY = 3147

REF_HOURLY_STAFF = {
    'weekday': {**{h: 1 for h in range(0, 8)},
                **{h: 6 for h in range(8, 14)},
                **{h: 7 for h in range(14, 20)},
                **{h: 4 for h in range(20, 24)}},
    'weekend': {**{h: 1 for h in range(0, 8)},
                **{h: 5 for h in range(8, 14)},
                **{h: 5 for h in range(14, 20)},
                **{h: 3 for h in range(20, 24)}},
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

REF_WD_DAILY = sum(REF_HOURLY_LAMBDA['weekday'].values())
REF_WE_DAILY = sum(REF_HOURLY_LAMBDA['weekend'].values())

# Shift presets
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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_shift_hours(start, end):
    """Get list of hours covered by a shift. Handles overnight (end < start)."""
    if end == start:
        return list(range(24))
    elif end > start:
        return list(range(start, end))
    else:
        return list(range(start, 24)) + list(range(0, end))


def format_hour(h):
    """Format hour as 12-hour AM/PM."""
    if h == 0 or h == 24:
        return "12:00 AM"
    elif h == 12:
        return "12:00 PM"
    elif h < 12:
        return f"{h}:00 AM"
    else:
        return f"{h-12}:00 PM"


def shift_label(s):
    """Generate display label for a shift."""
    duration = len(get_shift_hours(s['start'], s['end']))
    return f"{s['name']}  ({format_hour(s['start'])} – {format_hour(s['end'])}, {duration} hrs)"


# =============================================================================
# CORE MODEL ENGINE
# =============================================================================

def compute_hourly_requirements(monthly_opps, availability=0.60, sla_level='95%'):
    """Compute required staff per hour for weekday and weekend."""
    sla_factor = {'90%': 0.90, '95%': 1.00, '99%': 1.10}.get(sla_level, 1.00)
    avail_factor = 0.60 / availability
    volume_ratio = monthly_opps / REF_MONTHLY

    hourly_req = {}
    hourly_opps = {}
    for day_type in ['weekday', 'weekend']:
        for h in range(24):
            ref_staff = REF_HOURLY_STAFF[day_type][h]
            scaled = max(1, round(ref_staff * np.sqrt(volume_ratio) * sla_factor * avail_factor))
            hourly_req[(day_type, h)] = scaled
            hourly_opps[(day_type, h)] = REF_HOURLY_LAMBDA[day_type][h] * volume_ratio

    return hourly_req, hourly_opps


def solve_custom_shift_lp(shifts, monthly_opps, availability, sla_level,
                          max_opps_per_rep, max_days_per_week=6):
    """LP optimizer with custom shift definitions."""

    hourly_req, hourly_opps = compute_hourly_requirements(monthly_opps, availability, sla_level)
    volume_ratio = monthly_opps / REF_MONTHLY

    # Compute shift-level info
    shift_info = []
    for s in shifts:
        hours = get_shift_hours(s['start'], s['end'])
        wd_opps = sum(REF_HOURLY_LAMBDA['weekday'][h] for h in hours) * volume_ratio
        we_opps = sum(REF_HOURLY_LAMBDA['weekend'][h] for h in hours) * volume_ratio
        shift_info.append({
            **s,
            'hours': hours,
            'duration': len(hours),
            'wd_opps': round(wd_opps, 1),
            'we_opps': round(we_opps, 1),
            'wd_capacity_req': max(1, int(math.ceil(wd_opps / max_opps_per_rep))),
            'we_capacity_req': max(1, int(math.ceil(we_opps / max_opps_per_rep))),
        })

    # Generate day patterns
    day_patterns = {}
    for i in range(7):
        working = [DAYS[(i + j) % 7] for j in range(5)]
        day_patterns[f"5d_{i}"] = working
    if max_days_per_week >= 6:
        for i, off_day in enumerate(DAYS):
            day_patterns[f"6d_{i}"] = [d for d in DAYS if d != off_day]

    # Decision variables: x[pattern, shift_idx] = employees
    prob = LpProblem("Custom_Shift_Opt", LpMinimize)
    x = {}
    for dp_name in day_patterns:
        for si, s in enumerate(shift_info):
            var_name = f"x_{dp_name}__S{si}"
            x[(dp_name, si)] = LpVariable(var_name, lowBound=0, cat='Integer')

    # Objective: minimize total employees
    prob += lpSum(x[(dp, si)] for dp in day_patterns for si in range(len(shift_info)))

    # Constraint 1: Hourly coverage (SLA-driven)
    for d in DAYS:
        is_we = d in ['Saturday', 'Sunday']
        day_type = 'weekend' if is_we else 'weekday'
        for h in range(24):
            req = hourly_req[(day_type, h)]
            # Sum of staff from all shifts covering this hour, on patterns that include this day
            covering_vars = []
            for dp_name, work_days in day_patterns.items():
                if d in work_days:
                    for si, s in enumerate(shift_info):
                        if h in s['hours']:
                            covering_vars.append(x[(dp_name, si)])
            if covering_vars:
                prob += lpSum(covering_vars) >= req, f"Hour_{d}_{h}"

    # Constraint 2: Capacity per shift per day (max opps per rep)
    for d in DAYS:
        is_we = d in ['Saturday', 'Sunday']
        for si, s in enumerate(shift_info):
            cap_req = s['we_capacity_req'] if is_we else s['wd_capacity_req']
            shift_staff = []
            for dp_name, work_days in day_patterns.items():
                if d in work_days:
                    shift_staff.append(x[(dp_name, si)])
            if shift_staff:
                prob += lpSum(shift_staff) >= cap_req, f"Cap_{d}_S{si}"

    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        return None

    total = int(value(prob.objective))

    # Extract assignments
    assigned = []
    for dp_name, work_days in day_patterns.items():
        for si, s in enumerate(shift_info):
            val = x[(dp_name, si)].value()
            if val and val > 0:
                assigned.append({
                    'Count': int(val),
                    'Shift': s['name'],
                    'Timing': f"{format_hour(s['start'])} – {format_hour(s['end'])}",
                    'Duration': f"{s['duration']} hrs",
                    'Days/Week': len(work_days),
                    'Working Days': ', '.join(d[:3] for d in work_days),
                })
    assigned.sort(key=lambda r: (-r['Count'], r['Shift']))

    # Build hourly coverage matrix
    coverage = {}
    for d in DAYS:
        for h in range(24):
            covered = 0
            for dp_name, work_days in day_patterns.items():
                if d in work_days:
                    for si, s in enumerate(shift_info):
                        if h in s['hours']:
                            val = x[(dp_name, si)].value()
                            if val:
                                covered += int(val)
            coverage[(d, h)] = covered

    # Per-shift staffing on each day
    shift_by_day = {}
    for d in DAYS:
        for si, s in enumerate(shift_info):
            staff = 0
            for dp_name, work_days in day_patterns.items():
                if d in work_days:
                    val = x[(dp_name, si)].value()
                    if val:
                        staff += int(val)
            shift_by_day[(d, si)] = staff

    return {
        'total': total,
        'assigned': assigned,
        'coverage': coverage,
        'hourly_req': hourly_req,
        'hourly_opps': hourly_opps,
        'shift_info': shift_info,
        'shift_by_day': shift_by_day,
    }


def analyze_uploaded_data(df):
    """Extract arrival statistics from uploaded Opp Created data."""
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

    wd_hourly, we_hourly = {}, {}
    for h in range(24):
        wd_hourly[h] = ((~df['Is_Weekend']) & (df['Hour'] == h)).sum() / max(n_wd, 1)
        we_hourly[h] = ((df['Is_Weekend']) & (df['Hour'] == h)).sum() / max(n_we, 1)

    monthly = df.groupby('Year_Month').size().reset_index(name='Opportunities')
    monthly['Year_Month'] = monthly['Year_Month'].astype(str)

    hourly_chart = [{'Hour': f"{h:02d}:00", 'Weekday': wd_hourly[h], 'Weekend': we_hourly[h]} for h in range(24)]

    hourly_counts = df.groupby(['Date', 'Hour']).size().reset_index(name='count')
    disp = hourly_counts['count'].var() / hourly_counts['count'].mean() if hourly_counts['count'].mean() > 0 else 0

    avg_daily = len(df) / max(total_days, 1)

    return {
        'total_records': len(df),
        'date_range': (df['Opp Created'].min(), df['Opp Created'].max()),
        'total_days': total_days, 'n_weekdays': n_wd, 'n_weekends': n_we,
        'avg_daily': avg_daily,
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
                if uploaded.name.endswith('.csv'):
                    raw_df = pd.read_csv(uploaded, parse_dates=['Opp Created'])
                else:
                    raw_df = pd.read_excel(uploaded, parse_dates=['Opp Created'])
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

    availability = st.slider("Rep Availability", 0.40, 0.85, 0.60, 0.05,
                              help="Fraction of time reps are free for new opps")
    st.caption(f"Selected: {availability:.0%}")

    sla_level = st.selectbox("SLA Target", ['90%', '95%', '99%'], index=1)

    max_opps_per_rep = st.slider("Max Opportunities / Rep / Day", 3, 10, 6, 1,
                                  help="Capacity ceiling per rep per shift")

    max_days = st.radio("Max Days / Week", [5, 6], index=1,
                         format_func=lambda x: f"{x}-day work week")

    # --- Shift Configuration ---
    st.markdown("---")
    st.markdown("### Shift Configuration")

    num_shifts = st.selectbox("Number of Shifts", [3, 4, 5], index=1)

    use_preset = st.checkbox("Use preset configuration", value=True)

    shifts = []
    if use_preset:
        preset = SHIFT_PRESETS[num_shifts]
        st.caption(f"Preset: {preset['name']}")
        for s in preset['shifts']:
            st.caption(f"  {shift_label(s)}")
        shifts = preset['shifts']
    else:
        st.caption("Define each shift below:")
        for i in range(num_shifts):
            st.markdown(f"**Shift {i+1}**")
            c1, c2, c3 = st.columns(3)
            name = c1.text_input(f"Name", value=f"Shift {i+1}", key=f"sn_{i}",
                                  label_visibility="collapsed")
            start = c2.selectbox("Start", range(24), index=8,
                                  format_func=format_hour, key=f"ss_{i}")
            end = c3.selectbox("End", range(24), index=16,
                                format_func=format_hour, key=f"se_{i}")
            shifts.append({'name': name, 'start': start, 'end': end})
            duration = len(get_shift_hours(start, end))
            st.caption(f"  {format_hour(start)} – {format_hour(end)} ({duration} hrs)")

    st.markdown("---")
    st.caption("Post-restructuring: ~2,150/mo")
    st.caption("Pre-restructuring: ~3,500/mo")
    st.caption("Current team: 17 reps")


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown('<p class="main-header">B2C Sales Team Workforce Optimization</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Meydan Free Zone  |  Staffing Calculator</p>', unsafe_allow_html=True)

# --- Compute ---
result = solve_custom_shift_lp(shifts, monthly_opps, availability, sla_level,
                                max_opps_per_rep, max_days)

# --- Top Metrics ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Employees Required", result['total'] if result else "N/A")
m2.metric("Monthly Volume", f"{monthly_opps:,}")
m3.metric("Availability", f"{availability:.0%}")
m4.metric("SLA Target", sla_level)
m5.metric("Max Opps/Rep", max_opps_per_rep)

st.markdown("---")

# --- Shift Summary ---
with st.expander("Shift Definitions", expanded=True):
    if result:
        si_rows = []
        for s in result['shift_info']:
            si_rows.append({
                'Shift': s['name'],
                'Timing': f"{format_hour(s['start'])} – {format_hour(s['end'])}",
                'Duration': f"{s['duration']} hrs",
                'Weekday Opps': s['wd_opps'],
                'Weekend Opps': s['we_opps'],
                'WD Capacity Req': s['wd_capacity_req'],
                'WE Capacity Req': s['we_capacity_req'],
            })
        st.dataframe(pd.DataFrame(si_rows), use_container_width=True, hide_index=True)
        st.caption("Capacity Req = ceil(Shift Opps / Max Opps per Rep). "
                   "Final staffing is the higher of Capacity Req and SLA Req.")

# --- Tabs ---
tab_names = []
if data_stats:
    tab_names.append("Data Analysis")
tab_names += ["Shift Roster", "Coverage Matrix", "Hourly Heatmap", "Staffing Curve", "Scenarios"]
tabs = st.tabs(tab_names)
tab_idx = 0

# --- TAB: Data Analysis ---
if data_stats:
    with tabs[tab_idx]:
        st.subheader("Uploaded Data Analysis")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Records", f"{data_stats['total_records']:,}")
        d2.metric("Total Days", data_stats['total_days'])
        d3.metric("Avg Daily Volume", f"{data_stats['avg_daily']:.1f}")
        d4.metric("Est. Monthly", f"{data_stats['est_monthly']:,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Date Range:** {data_stats['date_range'][0].strftime('%Y-%m-%d')} to "
                        f"{data_stats['date_range'][1].strftime('%Y-%m-%d')}")
            st.markdown(f"**Weekdays:** {data_stats['n_weekdays']} days "
                        f"(avg {data_stats['avg_wd_daily']:.1f} opps/day)")
            st.markdown(f"**Weekends:** {data_stats['n_weekends']} days "
                        f"(avg {data_stats['avg_we_daily']:.1f} opps/day)")
        with c2:
            st.markdown(f"**Dispersion Index:** {data_stats['dispersion']:.2f} "
                        f"({'Bursty' if data_stats['dispersion'] > 1.5 else 'Poisson-like'})")
            st.markdown(f"**Weekday Peak:** {data_stats['peak_wd_hour']:02d}:00 "
                        f"({data_stats['wd_hourly'][data_stats['peak_wd_hour']]:.1f} opps/hr)")
            st.markdown(f"**Weekend Peak:** {data_stats['peak_we_hour']:02d}:00 "
                        f"({data_stats['we_hourly'][data_stats['peak_we_hour']]:.1f} opps/hr)")

        hc = data_stats['hourly_chart']
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(x=hc['Hour'], y=hc['Weekday'], name='Weekday',
                               marker_color='#2c3e50', opacity=0.85))
        fig_h.add_trace(go.Bar(x=hc['Hour'], y=hc['Weekend'], name='Weekend',
                               marker_color='#e74c3c', opacity=0.70))
        fig_h.update_layout(barmode='group', height=350,
                            margin=dict(l=40, r=20, t=20, b=40),
                            xaxis_title="Hour of Day", yaxis_title="Avg Arrivals / Hour",
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(fig_h, use_container_width=True)

        if len(data_stats['monthly_breakdown']) > 1:
            fig_m = px.bar(data_stats['monthly_breakdown'], x='Year_Month', y='Opportunities',
                           color_discrete_sequence=['#2c3e50'])
            fig_m.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=40),
                                xaxis_title="Month", yaxis_title="Opportunities")
            st.plotly_chart(fig_m, use_container_width=True)
    tab_idx += 1


# --- TAB: Shift Roster ---
with tabs[tab_idx]:
    st.subheader("Recommended Shift Assignments")
    if result and result['total']:
        roster_df = pd.DataFrame(result['assigned'])
        roster_df.index = range(1, len(roster_df) + 1)
        roster_df.index.name = '#'
        st.dataframe(roster_df, use_container_width=True)

        # Per-shift per-day staffing
        st.subheader("Staffing by Shift and Day")
        sbd = result['shift_by_day']
        sbd_rows = []
        for d in DAYS:
            row = {'Day': d + (' *' if d in ['Saturday', 'Sunday'] else '')}
            for si, s in enumerate(result['shift_info']):
                row[f"{s['name']} ({format_hour(s['start'])}-{format_hour(s['end'])})"] = sbd.get((d, si), 0)
            row['Total On Duty'] = sum(sbd.get((d, si), 0) for si in range(len(result['shift_info'])))
            sbd_rows.append(row)
        st.dataframe(pd.DataFrame(sbd_rows), use_container_width=True, hide_index=True)
        st.caption("* = Weekend")

        # Shift distribution chart
        st.subheader("Shift Distribution")
        s_df = pd.DataFrame(result['assigned'])
        s_agg = s_df.groupby(['Shift', 'Timing'])['Count'].sum().reset_index()
        fig_s = px.bar(s_agg, x='Shift', y='Count', text='Count',
                       hover_data=['Timing'],
                       color='Shift', color_discrete_sequence=px.colors.qualitative.Set2)
        fig_s.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=40),
                            showlegend=False, xaxis_title="", yaxis_title="Employees")
        fig_s.update_traces(textposition='outside')
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.error("No feasible solution. Try adjusting shift times or relaxing constraints.")
tab_idx += 1


# --- TAB: Coverage Matrix ---
with tabs[tab_idx]:
    st.subheader("Hourly Coverage: Scheduled vs Required")
    if result:
        cov = result['coverage']
        req = result['hourly_req']

        # Per-day coverage table (summarized by shift windows)
        st.markdown("**Daily Coverage Summary**")
        for day_type_label, day_list in [("Weekday (Monday)", ['Monday']), ("Weekend (Saturday)", ['Saturday'])]:
            d = day_list[0]
            is_we = d in ['Saturday', 'Sunday']
            dt = 'weekend' if is_we else 'weekday'
            st.markdown(f"*{day_type_label}*")
            cov_rows = []
            for h in range(24):
                r = req[(dt, h)]
                c = cov.get((d, h), 0)
                surplus = c - r
                shifts_covering = [s['name'] for s in result['shift_info'] if h in s['hours']]
                cov_rows.append({
                    'Hour': format_hour(h),
                    'Required': r,
                    'Scheduled': c,
                    'Surplus': f"+{surplus}" if surplus > 0 else ("=" if surplus == 0 else str(surplus)),
                    'Shifts Active': ', '.join(shifts_covering) if shifts_covering else 'None',
                })
            st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True, height=400)
tab_idx += 1


# --- TAB: Hourly Heatmap ---
with tabs[tab_idx]:
    st.subheader("Coverage Heatmap")
    if result:
        hm_type = st.radio("Show", ["Scheduled Staff", "Required Staff", "Surplus"],
                            horizontal=True, key="hm_radio")
        cov = result['coverage']
        req = result['hourly_req']

        hm_vals = []
        for d in DAYS:
            row = []
            dt = 'weekend' if d in ['Saturday', 'Sunday'] else 'weekday'
            for h in range(24):
                if hm_type == "Scheduled Staff":
                    row.append(cov.get((d, h), 0))
                elif hm_type == "Required Staff":
                    row.append(req[(dt, h)])
                else:
                    row.append(cov.get((d, h), 0) - req[(dt, h)])
            hm_vals.append(row)

        colorscale = 'RdYlGn' if hm_type == "Surplus" else 'Blues'
        fig_hm = go.Figure(data=go.Heatmap(
            z=hm_vals, x=HOUR_LABELS, y=[d[:3] for d in DAYS],
            text=[[str(v) for v in row] for row in hm_vals],
            texttemplate="%{text}", colorscale=colorscale,
            showscale=True, colorbar_title="Staff",
        ))

        # Add shift boundary annotations
        for s in result['shift_info']:
            fig_hm.add_vline(x=s['start'] - 0.5, line_dash="dot", line_color="rgba(0,0,0,0.3)",
                             annotation_text=s['name'], annotation_position="top",
                             annotation_font_size=10)

        fig_hm.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=30),
                             xaxis_title="Hour", yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_hm, use_container_width=True)
tab_idx += 1


# --- TAB: Staffing Curve ---
with tabs[tab_idx]:
    st.subheader("Headcount vs Monthly Volume")
    vols = list(range(500, 8001, 500))
    cr = []
    for v in vols:
        for av in [0.50, 0.60, 0.70]:
            r = solve_custom_shift_lp(shifts, v, av, sla_level, max_opps_per_rep, max_days)
            if r:
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
        st.caption("Dashed vertical: selected volume. Dotted red: current team (17).")
tab_idx += 1


# --- TAB: Scenario Comparison ---
with tabs[tab_idx]:
    st.subheader("Multi-Scenario Comparison")
    sv = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000]
    s_rows = []
    for v in sv:
        row = {'Volume': f"{v:,}"}
        for av in [0.50, 0.60, 0.70]:
            for sla in ['90%', '95%', '99%']:
                r = solve_custom_shift_lp(shifts, v, av, sla, max_opps_per_rep, max_days)
                row[f"{av:.0%} / {sla}"] = r['total'] if r else '-'
        s_rows.append(row)
    st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

    st.info(f"Current: **{monthly_opps:,}** opps/mo | **{availability:.0%}** avail | "
            f"**{sla_level}** SLA | **{max_opps_per_rep}** max opps/rep | "
            f"**{max_days}-day** week | **{num_shifts} shifts** | "
            f"**{result['total'] if result else 'N/A'} employees**")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

with st.expander("Model Methodology"):
    st.markdown("""
**Three-Layer Framework:** Erlang-C baseline (indicative) -> Monte Carlo simulation with 
empirical arrivals and bimodal service times (primary calibration) -> LP shift optimization 
with custom shift definitions (actionable roster).

**Custom Shifts:** The LP ensures that for every hour of every day, enough staff from 
overlapping shifts cover the hourly SLA requirement AND the per-shift capacity constraint 
(max opps/rep) is satisfied. Shifts can overlap, enabling staggered coverage that matches 
demand peaks more efficiently.

**Scaling:** Square-root staffing law extrapolates MC-calibrated hourly requirements to any 
volume level. Availability and SLA adjustments applied multiplicatively.

**Key Finding:** Current SLA failure (15.6%) is driven by assignment delay (median 6.6 hours), 
not headcount. Reps respond in 4 minutes once assigned.
    """)

with st.expander("Assumptions"):
    st.markdown(f"""
- **Service Time:** ~9 min blended (45% pickup x 12.5 min call + 55% x 2 min no-answer)
- **SLA Window:** 30 min from pool entry to first contact
- **Availability:** Set to **{availability:.0%}** — accounts for post-sale support, admin, breaks
- **Max Opps/Rep:** Set to **{max_opps_per_rep}** per rep per shift
- **Weekend:** Saturday/Sunday, ~60% of weekday volume
    """)

with st.expander("Input File Format"):
    st.markdown("""
Upload Excel (.xlsx) or CSV with one column: **Opp Created** (datetime).

| Opp Created |
|---|
| 2026-04-23 14:30:00 |
| 2026-04-23 15:12:00 |
    """)
