# =============================================================================
# B2C Sales Team Workforce Optimization — Streamlit Calculator
# =============================================================================
# Run: streamlit run streamlit_app.py
# Input: Excel/CSV with single column "Opp Created" (datetime)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import plotly.graph_objects as go
import plotly.express as px

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
    .main-header {
        font-size: 1.8rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #6c757d; margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CORE ENGINE
# =============================================================================

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
WINDOWS = ['Night', 'Morning', 'Afternoon', 'Evening']
WINDOW_HOURS = {
    'Night': list(range(0, 8)), 'Morning': list(range(8, 14)),
    'Afternoon': list(range(14, 20)), 'Evening': list(range(20, 24)),
}

# Time range labels
WINDOW_LABELS = {
    'Night':     '12:00 AM – 08:00 AM',
    'Morning':   '08:00 AM – 02:00 PM',
    'Afternoon': '02:00 PM – 08:00 PM',
    'Evening':   '08:00 PM – 12:00 AM',
}

SHIFT_LABELS = {
    'Night':     '12:00 AM – 08:00 AM  (8 hrs)',
    'Morning':   '08:00 AM – 02:00 PM  (6 hrs)',
    'Afternoon': '02:00 PM – 08:00 PM  (6 hrs)',
    'Evening':   '08:00 PM – 12:00 AM  (4 hrs)',
    'Day':       '08:00 AM – 08:00 PM  (12 hrs)',
}

# Pre-calibrated Monte Carlo baseline (from full assessment)
REF_MONTHLY = 3147
REF_MC = {
    'weekday': {'Night': 1, 'Morning': 6, 'Afternoon': 7, 'Evening': 4},
    'weekend': {'Night': 1, 'Morning': 5, 'Afternoon': 5, 'Evening': 3},
}

# Reference avg opps per window per day (from full dataset, used for capacity scaling)
REF_WINDOW_VOL = {
    'weekday': {'Night': 12.0, 'Morning': 45.2, 'Afternoon': 44.5, 'Evening': 15.0},
    'weekend': {'Night': 10.7, 'Morning': 25.6, 'Afternoon': 27.2, 'Evening': 11.7},
}
REF_WD_DAILY = sum(REF_WINDOW_VOL['weekday'].values())  # ~116.7
REF_WE_DAILY = sum(REF_WINDOW_VOL['weekend'].values())  # ~75.2

DEFAULT_MAX_OPPS_PER_REP = 6


def solve_shift_lp(requirements, max_days_per_week=6):
    """LP: minimize total employees subject to per-slot coverage."""
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
                for w in WINDOWS:
                    cov[(d, w)] = 1 if (d in work_days and w in covered) else 0
            patterns[combo] = {'coverage': cov, 'work_days': work_days,
                               'time_shift': ts_name, 'days_per_week': len(work_days)}

    prob = LpProblem("Shift_Opt", LpMinimize)
    x = {n: LpVariable(f"x_{n}", lowBound=0, cat='Integer') for n in patterns}
    prob += lpSum(x[n] for n in patterns)
    for d in DAYS:
        for w in WINDOWS:
            prob += lpSum(x[n] * patterns[n]['coverage'][(d, w)] for n in patterns) >= requirements[(d, w)]
    prob.solve(PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        return None

    total = int(value(prob.objective))
    assigned = []
    for n in patterns:
        val = x[n].value()
        if val and val > 0:
            shift_name = patterns[n]['time_shift']
            assigned.append({
                'Count': int(val), 'Shift': shift_name,
                'Timing': SHIFT_LABELS.get(shift_name, ''),
                'Days/Week': patterns[n]['days_per_week'],
                'Working Days': ', '.join(d[:3] for d in patterns[n]['work_days']),
            })
    assigned.sort(key=lambda r: -r['Count'])

    coverage = {}
    for d in DAYS:
        for w in WINDOWS:
            coverage[(d, w)] = int(sum(
                x[n].value() * patterns[n]['coverage'][(d, w)]
                for n in patterns if x[n].value() and x[n].value() > 0))

    return {'total': total, 'assigned': assigned, 'coverage': coverage, 'requirements': requirements}


def compute_staffing(monthly_opps, availability=0.60, sla_level='95%', max_days=6,
                     max_opps_per_rep=6):
    """Monthly volume -> headcount + shift roster with capacity constraint."""
    sla_factor = {'90%': 0.90, '95%': 1.00, '99%': 1.10}.get(sla_level, 1.00)
    avail_factor = 0.60 / availability
    volume_ratio = monthly_opps / REF_MONTHLY

    reqs = {}
    window_detail = {}
    capacity_detail = {}
    for d in DAYS:
        is_we = d in ['Saturday', 'Sunday']
        day_type = 'weekend' if is_we else 'weekday'
        for w in WINDOWS:
            # Queueing-based requirement (SLA-driven)
            ref = REF_MC[day_type][w]
            queueing_req = max(1, round(ref * np.sqrt(volume_ratio) * sla_factor * avail_factor))

            # Capacity-based requirement (max opps per rep)
            scaled_window_vol = REF_WINDOW_VOL[day_type][w] * volume_ratio
            capacity_req = max(1, int(np.ceil(scaled_window_vol / max_opps_per_rep)))

            # Final = max of both constraints
            final_req = max(queueing_req, capacity_req)
            reqs[(d, w)] = final_req
            window_detail[(day_type, w)] = final_req
            capacity_detail[(day_type, w)] = {
                'queueing': queueing_req, 'capacity': capacity_req,
                'binding': 'Capacity' if capacity_req > queueing_req else 'SLA',
                'window_vol': round(scaled_window_vol, 1),
            }

    sol = solve_shift_lp(reqs, max_days)
    return {
        'monthly_opps': monthly_opps, 'total': sol['total'] if sol else None,
        'per_window': window_detail, 'solution': sol,
        'volume_ratio': volume_ratio, 'capacity_detail': capacity_detail,
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

    # Hourly rates
    wd_hourly, we_hourly = {}, {}
    for h in range(24):
        wd_hourly[h] = ((~df['Is_Weekend']) & (df['Hour'] == h)).sum() / max(n_wd, 1)
        we_hourly[h] = ((df['Is_Weekend']) & (df['Hour'] == h)).sum() / max(n_we, 1)

    # Window rates
    wd_window, we_window = {}, {}
    for w, hours in WINDOW_HOURS.items():
        wd_window[w] = sum(wd_hourly[h] for h in hours) / len(hours)
        we_window[w] = sum(we_hourly[h] for h in hours) / len(hours)

    # Monthly breakdown
    monthly = df.groupby('Year_Month').size().reset_index(name='Opportunities')
    monthly['Year_Month'] = monthly['Year_Month'].astype(str)

    # Hourly chart data
    hourly_chart = []
    for h in range(24):
        hourly_chart.append({'Hour': f"{h:02d}:00", 'Weekday': wd_hourly[h], 'Weekend': we_hourly[h]})

    # Dispersion
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
        'wd_window': wd_window, 'we_window': we_window,
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
    st.markdown("### Data Input")
    st.markdown("---")

    input_mode = st.radio(
        "Input Method",
        ["Upload File", "Manual Entry"],
        help="Upload Excel/CSV with 'Opp Created' column, or enter volume manually"
    )

    data_stats = None
    monthly_opps = 2500

    if input_mode == "Upload File":
        uploaded = st.file_uploader(
            "Upload Opportunity Data",
            type=['xlsx', 'xls', 'csv'],
            help="Single column: 'Opp Created' (datetime)"
        )

        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    raw_df = pd.read_csv(uploaded, parse_dates=['Opp Created'])
                else:
                    raw_df = pd.read_excel(uploaded, parse_dates=['Opp Created'])

                if 'Opp Created' not in raw_df.columns:
                    st.error("Column 'Opp Created' not found. "
                             f"Available: {', '.join(raw_df.columns.tolist())}")
                else:
                    raw_df = raw_df[['Opp Created']].dropna()
                    raw_df['Opp Created'] = pd.to_datetime(raw_df['Opp Created'])
                    data_stats = analyze_uploaded_data(raw_df)
                    monthly_opps = int(round(data_stats['est_monthly']))
                    st.success(f"Loaded {data_stats['total_records']:,} records")
            except Exception as e:
                st.error(f"Error: {e}")

        monthly_opps = st.slider(
            "Monthly Volume (auto / override)",
            min_value=500, max_value=8000,
            value=min(max(monthly_opps, 500), 8000), step=100,
            help="Auto-filled from data. Adjust to model scenarios."
        )
    else:
        monthly_opps = st.slider(
            "Monthly Opportunity Volume",
            min_value=500, max_value=8000, value=2500, step=100,
        )

    st.markdown("---")
    st.markdown("### Model Parameters")

    availability = st.slider(
        "Rep Availability",
        min_value=0.40, max_value=0.85, value=0.60, step=0.05,
        help="Fraction of time reps are free for new opportunities"
    )
    st.caption(f"Selected: {availability:.0%}")

    sla_level = st.selectbox("SLA Target", ['90%', '95%', '99%'], index=1)

    max_opps_per_rep = st.slider(
        "Max Opportunities / Rep / Day",
        min_value=3, max_value=10, value=6, step=1,
        help="Maximum number of opportunities a single rep can handle per day. "
             "Acts as a capacity floor — if window volume exceeds this limit × reps, "
             "more staff are required regardless of SLA."
    )

    max_days = st.radio(
        "Max Days / Week", [5, 6], index=1,
        format_func=lambda x: f"{x}-day work week"
    )

    st.markdown("---")
    st.markdown("### Quick Reference")
    st.caption("Post-restructuring: ~2,150/mo")
    st.caption("Pre-restructuring: ~3,500/mo")
    st.caption("Current team: 17 reps")


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown('<p class="main-header">B2C Sales Team Workforce Optimization</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Meydan Free Zone  |  Staffing Calculator</p>', unsafe_allow_html=True)

result = compute_staffing(monthly_opps, availability, sla_level, max_days, max_opps_per_rep)

# --- Top Metrics ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Employees Required", result['total'])
m2.metric("Monthly Volume", f"{monthly_opps:,}")
m3.metric("Availability", f"{availability:.0%}")
m4.metric("SLA Target", sla_level)
m5.metric("Max Opps/Rep", max_opps_per_rep)

st.markdown("---")

# --- Shift Timing Reference ---
with st.expander("Shift & Window Timing Reference", expanded=False):
    ref_rows = []
    for shift, timing in SHIFT_LABELS.items():
        covers = "Morning + Afternoon" if shift == 'Day' else shift
        ref_rows.append({'Shift Type': shift, 'Timing': timing, 'Covers Window(s)': covers})
    st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

# --- Build Tabs ---
tab_names = []
if data_stats:
    tab_names.append("Data Analysis")
tab_names += ["Shift Roster", "Coverage Matrix", "Staffing Curve", "Scenario Comparison"]
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
            st.markdown("**Date Range**")
            st.write(f"{data_stats['date_range'][0].strftime('%Y-%m-%d')} to "
                     f"{data_stats['date_range'][1].strftime('%Y-%m-%d')}")
            st.markdown(f"**Weekdays:** {data_stats['n_weekdays']} days "
                        f"(avg {data_stats['avg_wd_daily']:.1f} opps/day)")
            st.markdown(f"**Weekends:** {data_stats['n_weekends']} days "
                        f"(avg {data_stats['avg_we_daily']:.1f} opps/day)")
        with c2:
            st.markdown("**Arrival Pattern**")
            st.markdown(f"Dispersion Index: **{data_stats['dispersion']:.2f}** "
                        f"({'Bursty' if data_stats['dispersion'] > 1.5 else 'Poisson-like'})")
            st.markdown(f"Weekday Peak: **{data_stats['peak_wd_hour']:02d}:00** "
                        f"({data_stats['wd_hourly'][data_stats['peak_wd_hour']]:.1f} opps/hr)")
            st.markdown(f"Weekend Peak: **{data_stats['peak_we_hour']:02d}:00** "
                        f"({data_stats['we_hourly'][data_stats['peak_we_hour']]:.1f} opps/hr)")

        # Hourly chart
        st.subheader("Hourly Arrival Pattern")
        hc = data_stats['hourly_chart']
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(x=hc['Hour'], y=hc['Weekday'], name='Weekday',
                               marker_color='#2c3e50', opacity=0.85))
        fig_h.add_trace(go.Bar(x=hc['Hour'], y=hc['Weekend'], name='Weekend',
                               marker_color='#e74c3c', opacity=0.70))
        fig_h.update_layout(
            barmode='group', height=380,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Hour of Day", yaxis_title="Avg Arrivals / Hour",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig_h, use_container_width=True)

        # Window rates
        st.subheader("Arrival Rates by Time Window")
        w_rows = []
        for w in WINDOWS:
            w_rows.append({
                'Window': w,
                'Hours': f"{WINDOW_HOURS[w][0]:02d}:00 - {WINDOW_HOURS[w][-1]+1:02d}:00",
                'Weekday (opps/hr)': round(data_stats['wd_window'][w], 2),
                'Weekend (opps/hr)': round(data_stats['we_window'][w], 2),
                'WE/WD Ratio': f"{data_stats['we_window'][w]/max(data_stats['wd_window'][w],0.01):.0%}",
            })
        st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)

        # Monthly trend
        if len(data_stats['monthly_breakdown']) > 1:
            st.subheader("Monthly Volume Trend")
            fig_m = px.bar(data_stats['monthly_breakdown'], x='Year_Month', y='Opportunities',
                           color_discrete_sequence=['#2c3e50'])
            fig_m.update_layout(height=300, margin=dict(l=40, r=20, t=20, b=40),
                                xaxis_title="Month", yaxis_title="Opportunities")
            st.plotly_chart(fig_m, use_container_width=True)

    tab_idx += 1

# --- TAB: Shift Roster ---
with tabs[tab_idx]:
    st.subheader("Recommended Shift Assignments")
    if result['solution']:
        roster_df = pd.DataFrame(result['solution']['assigned'])
        roster_df.index = range(1, len(roster_df) + 1)
        roster_df.index.name = '#'
        st.dataframe(roster_df, use_container_width=True)

        st.subheader("Per-Window Staffing Requirements")
        pw = result['per_window']
        cd = result.get('capacity_detail', {})
        pw_rows = []
        for dt in ['weekday', 'weekend']:
            row = {'Day Type': dt.capitalize()}
            for w in WINDOWS:
                col_label = f"{w} ({WINDOW_LABELS[w]})"
                staff = pw.get((dt, w), 0)
                detail = cd.get((dt, w), {})
                binding = detail.get('binding', '')
                vol = detail.get('window_vol', 0)
                row[col_label] = f"{staff}  [{binding}]"
            pw_rows.append(row)
        st.dataframe(pd.DataFrame(pw_rows), use_container_width=True, hide_index=True)
        st.caption("Binding constraint: [SLA] = queueing/response time driven | "
                   "[Capacity] = max opps per rep driven")

        # Detailed breakdown table
        st.subheader("Constraint Breakdown")
        bd_rows = []
        for dt in ['weekday', 'weekend']:
            for w in WINDOWS:
                detail = cd.get((dt, w), {})
                bd_rows.append({
                    'Day Type': dt.capitalize(),
                    'Window': f"{w} ({WINDOW_LABELS[w]})",
                    'Avg Opps/Window': detail.get('window_vol', 0),
                    'SLA Requirement': detail.get('queueing', 0),
                    'Capacity Requirement': detail.get('capacity', 0),
                    'Final (max)': pw.get((dt, w), 0),
                    'Binding': detail.get('binding', ''),
                })
        st.dataframe(pd.DataFrame(bd_rows), use_container_width=True, hide_index=True)

        st.subheader("Shift Distribution")
        s_df = pd.DataFrame(result['solution']['assigned'])
        s_df['Shift Label'] = s_df['Shift'].map(lambda s: f"{s}\n{SHIFT_LABELS.get(s, '')}")
        fig_s = px.bar(s_df, x='Shift Label', y='Count', color='Shift', text='Count',
                       color_discrete_map={'Day': '#2c3e50', 'Morning': '#3498db',
                                           'Afternoon': '#e67e22', 'Evening': '#9b59b6',
                                           'Night': '#34495e'})
        fig_s.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=40),
                            showlegend=False, xaxis_title="", yaxis_title="Employees")
        fig_s.update_traces(textposition='outside')
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.error("No feasible solution. Try relaxing constraints.")
tab_idx += 1

# --- TAB: Coverage Matrix ---
with tabs[tab_idx]:
    st.subheader("Weekly Coverage: Scheduled vs Required")
    if result['solution']:
        cov = result['solution']['coverage']
        req = result['solution']['requirements']
        c_rows = []
        for d in DAYS:
            row = {'Day': d + (' *' if d in ['Saturday', 'Sunday'] else '')}
            for w in WINDOWS:
                c_val, r_val = cov[(d, w)], req[(d, w)]
                delta = c_val - r_val
                col_label = f"{w} ({WINDOW_LABELS[w]})"
                row[col_label] = f"{c_val} / {r_val}" + (f" (+{delta})" if delta > 0 else "")
            c_rows.append(row)
        st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)
        st.caption("Format: Scheduled / Required. * = Weekend")

        st.subheader("Coverage Heatmap")
        hm_vals = [[cov[(d, w)] for w in WINDOWS] for d in DAYS]
        hm_x_labels = [f"{w}\n{WINDOW_LABELS[w]}" for w in WINDOWS]
        fig_hm = go.Figure(data=go.Heatmap(
            z=hm_vals, x=hm_x_labels, y=[d[:3] for d in DAYS],
            text=[[str(v) for v in row] for row in hm_vals],
            texttemplate="%{text}", colorscale='Blues',
            showscale=True, colorbar_title="Reps",
        ))
        fig_hm.update_layout(height=350, margin=dict(l=40, r=40, t=20, b=30),
                             xaxis_title="Time Window", yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_hm, use_container_width=True)
tab_idx += 1

# --- TAB: Staffing Curve ---
with tabs[tab_idx]:
    st.subheader("Headcount vs Monthly Volume")
    vols = list(range(500, 8001, 250))
    cr = []
    for v in vols:
        for av in [0.50, 0.60, 0.70]:
            r = compute_staffing(v, av, sla_level, max_days, max_opps_per_rep)
            cr.append({'Monthly Volume': v, 'Availability': f"{av:.0%}", 'Headcount': r['total']})
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
                r = compute_staffing(v, av, sla, max_days, max_opps_per_rep)
                row[f"{av:.0%} / {sla}"] = r['total']
        s_rows.append(row)
    st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

    st.info(f"Current selection: **{monthly_opps:,}** opps/mo | "
            f"**{availability:.0%}** availability | **{sla_level}** SLA | "
            f"**{max_days}-day** week | Max **{max_opps_per_rep}** opps/rep | "
            f"**{result['total']} employees**")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

with st.expander("Model Methodology"):
    st.markdown("""
**Three-Layer Framework:** Erlang-C analytical baseline (indicative) → Monte Carlo simulation 
with empirical arrivals and bimodal service times (primary, pre-calibrated) → LP shift 
optimization (actionable roster). Scaling uses square-root staffing law.

**Key Finding:** Current SLA failure (15.6%) is driven by assignment delay (median 6.6 hours), 
not headcount. Reps respond in 4 minutes once assigned.
    """)

with st.expander("Assumptions"):
    st.markdown(f"""
- **Service Time:** ~9 min blended (45% pickup x 12.5 min call + 55% x 2 min no-answer)
- **SLA Window:** 30 min from pool entry to first contact
- **Availability:** Accounts for post-sale support (~30%), admin, breaks
- **Weekend:** Saturday/Sunday, ~60% of weekday volume
- **Max Opps/Rep:** Currently set to **{max_opps_per_rep}** per rep per window — acts as a capacity floor alongside the SLA-driven queueing requirement
    """)

with st.expander("Input File Format"):
    st.markdown("""
Upload Excel (.xlsx) or CSV with one column:

| Opp Created |
|---|
| 2026-04-23 14:30:00 |
| 2026-04-23 15:12:00 |

Column must be named **Opp Created** (datetime). All other columns are ignored.
    """)
