"""
Deep Excavation Analysis Tool - Web Application
Version 0.4

Modules:
  5. Anchored Wall Analysis
  6. Sheet Pile Section Library
  7A. Cantilever Wall (Free Earth + Blum)
  9. Staged Excavation Analysis

Run: python -m streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math
import io
import os
import sys
from datetime import date

# ── Engine imports ──
sys.path.insert(0, os.path.dirname(__file__))
from engine.models import ProjectInput, SoilLayer, WaterTable, SoilType, Surcharge, SurchargeType
from engine.anchored_wall import (
    Anchor, analyze_anchored_wall, get_wall_pressure_distribution,
)
from engine.section_library import (
    get_all_sections, get_section_by_name, get_sections_by_type,
    get_sections_by_manufacturer, get_manufacturers, search_sections,
    ProfileType, SteelGrade, STEEL_FY,
    check_section, check_section_wsd, auto_select, compare_sections,
    get_grade_comparison, database_summary, SheetPileSection,
)
from engine.cantilever_wall import (
    analyze_cantilever_free_earth, analyze_cantilever_blum,
    analyze_cantilever_both, cantilever_design_table, CantileverMethod,
)
from plots import (
    plot_soil_profile, plot_internal_forces, plot_pressure_distribution,
    plot_net_pressure, plot_utilization, plot_combined_crosssection,
)
from plots_staged import plot_staged_envelope, plot_stage_summary_bars
from engine.staged_excavation import (
    analyze_staged_excavation, generate_stages, ConstructionStage,
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Deep Excavation Tool",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Base font ── */
    html, body, [class*="css"] {
        font-size: 17px;
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    /* ── Sidebar: 30-35% width ── */
    [data-testid="stSidebar"] {
        min-width: 30vw !important;
        max-width: 35vw !important;
        background-color: #f8f9fb;
    }
    [data-testid="stSidebar"] > div { font-size: 15px; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stTextInput label {
        font-size: 15px;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-size: 16px;
        font-weight: 500;
    }
    /* Metric cards */
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    div[data-testid="stMetricLabel"] { font-size: 1.0rem; }
    /* Headers */
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.8rem !important; }
    h3 { font-size: 1.4rem !important; }
    h4 { font-size: 1.2rem !important; }
    /* Table / dataframe */
    .stDataFrame, .stDataFrame td, .stDataFrame th { font-size: 15px !important; }
    /* Expander titles */
    .streamlit-expanderHeader { font-size: 16px !important; font-weight: 600; }
    /* Radio buttons */
    [data-testid="stSidebar"] .stRadio label { font-size: 16px; }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        font-size: 15px; padding: 5px 0;
    }
    /* Buttons */
    .stButton > button { font-size: 16px; }
    /* Caption */
    .stCaption { font-size: 14px !important; }
    /* Number inputs / text inputs */
    .stNumberInput input, .stTextInput input { font-size: 15px !important; }
    .stSelectbox [data-baseweb="select"] { font-size: 15px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        'project_name': 'Project ABC',
        'location': 'Mumbai',
        'excavation_depth': 6.0,
        'surcharge': 10.0,
        'n_layers': 2,
        'layers': [
            {'name': 'Layer 1', 'top': 0.0, 'bottom': 4.0, 'gamma': 18.0, 'phi': 30.0, 'c': 0.0},
            {'name': 'Layer 2', 'top': 4.0, 'bottom': 12.0, 'gamma': 20.0, 'phi': 35.0, 'c': 10.0},
        ],
        'gwt_behind': 3.0,
        'gwt_front': 6.0,
        'wall_toe': 8.0,
        'wall_EI': 50000.0,
        'wall_section': 'LARSSEN 600',
        'n_anchors': 1,
        'anchors': [
            {'level': 1.5, 'type': 'rebar', 'incl': 20.0, 'spacing': 3.0,
             'bond_stress': 200.0, 'drill_dia': 115.0,
             'rebar_dia': 25.0, 'rebar_fy': 500.0, 'rebar_count': 1,
             'sda_size': 'R32', 'tendon_type': 'strand', 'prestress_ratio': 0.6},
        ],
        'wind_barrier': True,
        'barrier_height': 6.0,
        'Vb': 44.0,
        'terrain_cat': 3,
        'spring_model': False,
        'anchor_k': 5.0,
        'toe_k': 50.0,
        'result': None,
        'result_wind': None,
        'analysis_mode': 'Anchored Wall',
        'engineer_name': 'Engr. ABC',
        'firm_name': 'ABC Consultants',
        'revision': 'R0',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def compute_wind_load(h, Vb, terrain_cat):
    k2_table = {2: {5: 1.00, 10: 1.05}, 3: {5: 0.91, 10: 0.97}, 4: {5: 0.80, 10: 0.80}}
    k2_data = k2_table.get(terrain_cat, k2_table[3])
    heights = sorted(k2_data.keys())
    if h <= heights[0]: k2 = k2_data[heights[0]]
    elif h >= heights[-1]: k2 = k2_data[heights[-1]]
    else:
        for i in range(len(heights)-1):
            if heights[i] <= h <= heights[i+1]:
                r = (h - heights[i]) / (heights[i+1] - heights[i])
                k2 = k2_data[heights[i]] + r * (k2_data[heights[i+1]] - k2_data[heights[i]])
                break
    Vz = Vb * 1.0 * k2 * 1.0
    pz = 0.6 * Vz**2 / 1000
    F = 1.2 * pz * h
    M = F * h / 2
    return F, M, pz, Vz, k2


def build_project(exc_override=None):
    layers = []
    for lay in st.session_state.layers:
        thickness = lay['bottom'] - lay['top']
        gamma = lay['gamma']
        gamma_sat = gamma + 2.0
        phi = lay['phi']
        c = lay['c']
        stype = SoilType.CLAY if c > 0 and phi < 5 else (SoilType.SAND if c == 0 else SoilType.MIXED)
        layers.append(SoilLayer(
            name=lay['name'], thickness=thickness,
            gamma=gamma, gamma_sat=gamma_sat,
            c_eff=min(c, 5.0), phi_eff=phi, c_u=c, soil_type=stype,
        ))
    wt = WaterTable(
        depth_behind_wall=st.session_state.gwt_behind,
        depth_in_excavation=st.session_state.gwt_front,
    )
    surcharges = []
    if st.session_state.surcharge > 0:
        surcharges.append(Surcharge(surcharge_type=SurchargeType.UNIFORM, magnitude=st.session_state.surcharge))

    exc = exc_override or st.session_state.excavation_depth
    total_soil = sum(l.thickness for l in layers)
    min_required = exc + 5.0
    if total_soil < min_required:
        extra = min_required - total_soil + 2.0
        layers[-1] = SoilLayer(
            name=layers[-1].name, thickness=layers[-1].thickness + extra,
            gamma=layers[-1].gamma, gamma_sat=layers[-1].gamma_sat,
            c_eff=layers[-1].c_eff, phi_eff=layers[-1].phi_eff,
            c_u=layers[-1].c_u, soil_type=layers[-1].soil_type,
        )

    return ProjectInput(
        name=st.session_state.project_name,
        excavation_depth=exc, soil_layers=layers,
        water_table=wt, surcharges=surcharges,
    )


def build_anchors():
    anchors = []
    for i, a in enumerate(st.session_state.anchors):
        anc = Anchor(
            level=a['level'], anchor_type=a['type'], inclination=a['incl'],
            horizontal_spacing=a['spacing'], bond_stress=a['bond_stress'],
            drill_diameter=a['drill_dia'] / 1000, label=f"Anchor-{i+1} ({a['level']}m)",
        )
        if a['type'] == 'rebar':
            anc.rebar_dia = a['rebar_dia']; anc.rebar_fy = a['rebar_fy']; anc.rebar_count = a['rebar_count']
        elif a['type'] == 'sda':
            anc.sda_size = a['sda_size']
        elif a['type'] == 'prestressed':
            anc.tendon_type = a['tendon_type']; anc.prestress_ratio = a['prestress_ratio']
        anchors.append(anc)
    return anchors


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("Deep Excavation Tool")
    st.markdown("---")

    # ── Analysis Mode Selector ──
    modes = ["Anchored Wall", "Cantilever Wall", "Staged Excavation", "Section Library"]
    st.session_state.analysis_mode = st.radio(
        "**Analysis Module**", modes,
        index=modes.index(st.session_state.get('analysis_mode', 'Anchored Wall')),
        key="mode_radio"
    )
    st.markdown("---")

    # ── Common: Project & Soil (not shown for Section Library) ──
    if st.session_state.analysis_mode != "Section Library":
        with st.expander("Project Information", expanded=False):
            st.session_state.project_name = st.text_input("Project Name", st.session_state.project_name)
            st.session_state.location = st.text_input("Location", st.session_state.location)
            st.session_state.excavation_depth = st.number_input(
                "Excavation Depth (m)", 2.0, 30.0, st.session_state.excavation_depth, 0.5)
            st.session_state.surcharge = st.number_input(
                "Surcharge at GL (kPa)", 0.0, 50.0, st.session_state.surcharge, 5.0)
            st.markdown("---")
            if 'engineer_name' not in st.session_state:
                st.session_state.engineer_name = ''
            if 'firm_name' not in st.session_state:
                st.session_state.firm_name = ''
            if 'revision' not in st.session_state:
                st.session_state.revision = 'R0'
            st.session_state.engineer_name = st.text_input("Engineer Name", st.session_state.engineer_name)
            st.session_state.firm_name = st.text_input("Firm / Company", st.session_state.firm_name)
            st.session_state.revision = st.text_input("Revision", st.session_state.revision)

        with st.expander("Soil Profile", expanded=True):
            st.session_state.gwt_behind = st.number_input(
                "GWT behind wall (m)", 0.0, 30.0, st.session_state.gwt_behind, 0.5)
            st.session_state.gwt_front = st.number_input(
                "GWT in front (m)", 0.0, 30.0, st.session_state.gwt_front, 0.5)
            st.markdown("**Soil Layers** (top to bottom)")
            n_lay = st.number_input("Number of layers", 1, 8, len(st.session_state.layers), 1)
            while len(st.session_state.layers) < n_lay:
                prev = st.session_state.layers[-1]
                st.session_state.layers.append({
                    'name': f'Layer {len(st.session_state.layers)+1}',
                    'top': prev['bottom'], 'bottom': prev['bottom'] + 2.0,
                    'gamma': 18.0, 'phi': 30.0, 'c': 0.0})
            while len(st.session_state.layers) > n_lay:
                st.session_state.layers.pop()
            for i, lay in enumerate(st.session_state.layers):
                st.markdown(f"**Layer {i+1}**")
                c1, c2 = st.columns(2)
                lay['name'] = c1.text_input(f"Name##L{i}", lay['name'], key=f"ln_{i}")
                lay['gamma'] = c2.number_input(f"γ (kN/m³)##L{i}", 10.0, 28.0, lay['gamma'], 0.5, key=f"lg_{i}")
                c3, c4, c5 = st.columns(3)
                lay['top'] = c3.number_input(f"Top (m)##L{i}", 0.0, 50.0, lay['top'], 0.5, key=f"lt_{i}")
                lay['bottom'] = c4.number_input(f"Bot (m)##L{i}", 0.5, 50.0, lay['bottom'], 0.5, key=f"lb_{i}")
                lay['phi'] = c5.number_input(f"φ (°)##L{i}", 0.0, 45.0, lay['phi'], 1.0, key=f"lp_{i}")
                lay['c'] = st.number_input(f"c (kPa)##L{i}", 0.0, 500.0, lay['c'], 5.0, key=f"lc_{i}")



def _page_caption(extra=""):
    parts = [st.session_state.location, f"Excavation: {st.session_state.excavation_depth}m"]
    if extra:
        parts.append(extra)
    firm = st.session_state.get('firm_name', '')
    eng = st.session_state.get('engineer_name', '')
    rev = st.session_state.get('revision', '')
    if firm:
        parts.append(firm)
    if eng:
        parts.append(f"Engr: {eng}")
    if rev:
        parts.append(rev)
    return " | ".join(parts)


# ══════════════════════════════════════════════════════════════
# PAGE: SECTION LIBRARY
# ══════════════════════════════════════════════════════════════
if st.session_state.analysis_mode == "Section Library":
    st.markdown("## 🔩 Sheet Pile Section Library")
    summary = database_summary()
    st.caption(f"{summary['total_sections']} sections | {summary['steel_grades']} steel grades | "
               f"{', '.join(get_manufacturers())}")

    tab_browse, tab_check, tab_auto = st.tabs(["📋 Browse", "🔍 Check Section", "⚡ Auto-Select"])

    # ── Browse ──
    with tab_browse:
        col_f1, col_f2 = st.columns(2)
        mfr_filter = col_f1.selectbox("Manufacturer", ["All"] + get_manufacturers())
        type_filter = col_f2.selectbox("Profile Type", ["All"] + [pt.value for pt in ProfileType])

        sections = get_all_sections()
        if mfr_filter != "All":
            sections = [s for s in sections if s.manufacturer == mfr_filter]
        if type_filter != "All":
            sections = [s for s in sections if s.profile_type.value == type_filter]

        search_q = st.text_input("Search by name", "")
        if search_q:
            sections = [s for s in sections if search_q.upper() in s.name.upper()]

        st.caption(f"Showing {len(sections)} sections")
        rows = []
        for s in sections:
            rows.append({
                "Name": s.name, "Type": s.profile_type.value, "Mfr": s.manufacturer,
                "h (mm)": s.height, "t (mm)": s.thickness_web,
                "Wt (kg/m²)": s.weight,
                "Ze (cm³/m)": s.elastic_modulus, "Zp (cm³/m)": s.plastic_modulus,
                "I (cm⁴/m)": s.moment_of_inertia,
                "EI (kN·m²/m)": f"{s.EI_per_m:.0f}",
            })
        if rows:
            st.dataframe(rows, use_container_width=True, height=500)

    # ── Check Section ──
    with tab_check:
        st.subheader("Utilization Check")
        cc1, cc2 = st.columns(2)
        all_sec_names = [s.name for s in get_all_sections()]
        sec_name = cc1.selectbox("Section", all_sec_names,
            index=min(6, len(all_sec_names)-1), key="chk_sec")
        grade_name = cc2.selectbox("Steel Grade", [g.value for g in SteelGrade], index=2, key="chk_grade")

        cf1, cf2, cf3 = st.columns(3)
        M_chk = cf1.number_input("BM (kN·m/m)", 0.0, 5000.0, 100.0, 10.0, key="chk_m")
        V_chk = cf2.number_input("SF (kN/m)", 0.0, 2000.0, 50.0, 10.0, key="chk_v")
        P_chk = cf3.number_input("Axial (kN/m)", 0.0, 2000.0, 0.0, 10.0, key="chk_p")

        method_chk = st.radio("Method", ["LSM (IS 800:2007)", "WSD (IS 800:1984)"],
                              horizontal=True, key="chk_method")

        if st.button("Check", type="primary", key="chk_btn"):
            sec = get_section_by_name(sec_name)
            grade = SteelGrade(grade_name)
            fn = check_section if "LSM" in method_chk else check_section_wsd
            res = fn(sec, grade, M_chk, V_chk, P_chk)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Bending", f"{res.utilization_bending:.1%}", res.status_bending)
            m2.metric("Shear", f"{res.utilization_shear:.1%}", res.status_shear)
            m3.metric("Combined", f"{res.utilization_combined:.1%}", res.status_combined)
            m4.metric("Interlock", f"{res.utilization_interlock:.1%}", res.status_interlock)

            status_icon = '✅' if res.overall_status == 'OK' else '❌'
            st.markdown(f"**Governing: {res.governing_check} = {res.max_utilization:.1%} — "
                       f"{status_icon} {res.overall_status}**")
            st.markdown(f"Md = {res.Md:.1f} kN·m/m | Vd = {res.Vd:.1f} kN/m | fy = {STEEL_FY[grade]} MPa")

            # Bar chart
            fig = plot_utilization(res, sec_name, grade_name)
            st.plotly_chart(fig, use_container_width=True)

            # Grade comparison
            st.markdown("#### Grade Comparison")
            gc = get_grade_comparison(sec_name, M_chk, V_chk, P_chk,
                                     method="LSM" if "LSM" in method_chk else "WSD")
            gc_rows = []
            for r in gc:
                gc_rows.append({
                    "Grade": r.grade.value, "fy (MPa)": STEEL_FY[r.grade],
                    "Md (kN·m/m)": f"{r.Md:.1f}",
                    "Util": f"{r.utilization_bending:.1%}",
                    "Status": r.overall_status,
                })
            st.dataframe(gc_rows, use_container_width=True)

    # ── Auto-Select ──
    with tab_auto:
        st.subheader("Auto-Select Lightest Adequate Section")
        ac1, ac2 = st.columns(2)
        M_auto = ac1.number_input("Design BM (kN·m/m)", 0.0, 5000.0, 200.0, 10.0, key="auto_m")
        V_auto = ac2.number_input("Design SF (kN/m)", 0.0, 2000.0, 100.0, 10.0, key="auto_v")

        ac3, ac4, ac5 = st.columns(3)
        grade_auto = SteelGrade(ac3.selectbox("Grade", [g.value for g in SteelGrade], index=2, key="auto_g"))
        types_sel = ac4.multiselect("Profile Types", [pt.value for pt in ProfileType],
                                     default=["Z-profile", "U-profile"], key="auto_types")
        max_util = ac5.number_input("Max Utilization", 0.5, 1.0, 0.90, 0.05, key="auto_util")

        if st.button("Find Optimal Section", type="primary", key="auto_btn"):
            ptypes = [pt for pt in ProfileType if pt.value in types_sel] if types_sel else None
            sel = auto_select(M_auto, V_auto, grade=grade_auto, profile_types=ptypes,
                             max_utilization=max_util)

            status = '✅' if sel.utilization.overall_status == 'OK' else '⚠️'
            st.success(f"{status} **{sel.recommended.name}** ({sel.recommended.manufacturer}) — "
                      f"{sel.recommended.weight:.0f} kg/m², util = {sel.utilization.max_utilization:.1%}")
            st.markdown(f"EI = {sel.recommended.EI_per_m:.0f} kN·m²/m | "
                       f"Zp = {sel.recommended.plastic_modulus} cm³/m | "
                       f"Md = {sel.utilization.Md:.0f} kN·m/m")

            if sel.alternatives:
                st.markdown("#### Alternatives")
                alt_rows = []
                for sec_a, res_a in sel.alternatives:
                    alt_rows.append({
                        "Section": sec_a.name, "Mfr": sec_a.manufacturer,
                        "Type": sec_a.profile_type.value,
                        "Wt (kg/m²)": sec_a.weight,
                        "Util": f"{res_a.max_utilization:.1%}",
                        "Md": f"{res_a.Md:.0f}",
                        "EI": f"{sec_a.EI_per_m:.0f}",
                    })
                st.dataframe(alt_rows, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: CANTILEVER WALL
# ══════════════════════════════════════════════════════════════
elif st.session_state.analysis_mode == "Cantilever Wall":
    st.markdown(f"## {st.session_state.project_name} — Cantilever Wall Analysis")
    st.caption(_page_caption())

    tab_cfg, tab_res, tab_plt, tab_tbl, tab_rpt = st.tabs([
        "Configuration", "Results", "Diagrams", "Design Table", "Report"
    ])

    # ── Configuration ──
    with tab_cfg:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Wall Section")
            all_secs = get_all_sections()
            sec_names = [s.name for s in all_secs]
            default_idx = sec_names.index(st.session_state.wall_section) if st.session_state.wall_section in sec_names else 0
            cant_section = st.selectbox("Sheet Pile Section", sec_names, index=default_idx, key="cant_sec")
            sec_obj = get_section_by_name(cant_section)
            if sec_obj:
                cant_EI = sec_obj.EI_per_m
                st.caption(f"h = {sec_obj.height} mm | Zp = {sec_obj.plastic_modulus} cm³/m | "
                          f"EI = {cant_EI:.0f} kN·m²/m | {sec_obj.manufacturer}")
            else:
                cant_EI = 50000.0

            cant_grade = SteelGrade(st.selectbox("Steel Grade",
                [g.value for g in SteelGrade], index=2, key="cant_grade"))

        with c2:
            st.markdown("### Design Parameters")
            fos_passive = st.number_input("FOS on passive (IS 9527: 1.5–2.0)", 1.0, 3.0, 1.5, 0.1, key="cant_fos")
            toe_kick = st.number_input("Toe kick factor", 1.0, 1.5, 1.2, 0.05, key="cant_kick")
            cant_method = st.radio("Method",
                ["Both — compare", "Free Earth Support", "Blum's Fixed Earth"],
                horizontal=True, key="cant_method")

        if st.button("🚀 Run Cantilever Analysis", type="primary", use_container_width=True, key="cant_run"):
            try:
                project = build_project()
                if "Both" in cant_method:
                    fe, bl = analyze_cantilever_both(project, fos_passive, toe_kick, EI=cant_EI)
                    st.session_state.cant_fe = fe
                    st.session_state.cant_bl = bl
                elif "Free" in cant_method:
                    st.session_state.cant_fe = analyze_cantilever_free_earth(project, fos_passive, toe_kick, EI=cant_EI)
                    st.session_state.cant_bl = None
                else:
                    st.session_state.cant_fe = None
                    st.session_state.cant_bl = analyze_cantilever_blum(project, fos_passive, toe_kick, EI=cant_EI)
                st.session_state.cant_sec_obj = sec_obj
                st.session_state.cant_grade_val = cant_grade
                st.success("Analysis complete — see Results tab.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # ── Results ──
    with tab_res:
        res_fe = st.session_state.get('cant_fe')
        res_bl = st.session_state.get('cant_bl')
        if not res_fe and not res_bl:
            st.info("Run the analysis from the Configuration tab first.")
        else:
            items = []
            if res_fe: items.append(("Free Earth Support", res_fe))
            if res_bl: items.append(("Blum's Fixed Earth", res_bl))

            # ── Comparison table if both ──
            if len(items) == 2:
                st.subheader("Method Comparison")
                cols = st.columns(2)
                for idx, (label, res) in enumerate(items):
                    with cols[idx]:
                        st.markdown(f"#### {label}")
                        st.metric("Embedment required", f"{res.embedment_depth:.2f} m")
                        st.metric("Embedment with kick", f"{res.embedment_with_fos:.2f} m")
                        st.metric("Total wall length", f"{res.total_wall_length:.2f} m")
                        st.metric("Max BM", f"{res.max_bm:.1f} kN·m/m",
                                 f"at {res.max_bm_depth:.1f}m depth")
                        st.metric("Max SF", f"{res.max_sf:.1f} kN/m")
                        st.metric("Top deflection", f"{res.deflections[0]:.1f} mm")
                        fos_color = "normal" if res.fos_moment >= 1.5 else "inverse"
                        st.metric("FOS (moment)", f"{res.fos_moment:.2f}",
                                 "≥ 1.5 ✅" if res.fos_moment >= 1.5 else "< 1.5 ⚠️",
                                 delta_color=fos_color)
            else:
                label, res = items[0]
                st.subheader(f"{label} — Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Embedment", f"{res.embedment_with_fos:.2f} m")
                m2.metric("Total Length", f"{res.total_wall_length:.2f} m")
                m3.metric("Max BM", f"{res.max_bm:.1f} kN·m/m")
                m4.metric("FOS (moment)", f"{res.fos_moment:.2f}")
                st.metric("Top Deflection", f"{res.deflections[0]:.1f} mm")

            # ── Section adequacy ──
            st.markdown("---")
            st.subheader("Sheet Pile Adequacy")
            sec_obj = st.session_state.get('cant_sec_obj')
            grade = st.session_state.get('cant_grade_val', SteelGrade.S355GP)
            governing = res_fe if res_fe else res_bl
            if sec_obj and governing:
                util = check_section(sec_obj, grade, governing.max_bm, governing.max_sf)
                uc1, uc2, uc3, uc4 = st.columns(4)
                status_icon = '✅' if util.overall_status == 'OK' else '❌'
                uc1.metric(f"{sec_obj.name}", f"{util.max_utilization:.1%}", f"{status_icon} {util.overall_status}")
                uc2.metric("Md", f"{util.Md:.0f} kN·m/m")
                uc3.metric("Vd", f"{util.Vd:.0f} kN/m")
                uc4.metric("fy", f"{STEEL_FY[grade]} MPa")

                # Recommendations
                if util.max_utilization > 1.0:
                    sel = auto_select(governing.max_bm, governing.max_sf, grade=grade)
                    st.error(f"**Section inadequate.** Minimum: **{sel.recommended.name}** "
                            f"({sel.recommended.manufacturer}, {sel.recommended.weight:.0f} kg/m², "
                            f"util = {sel.utilization.max_utilization:.1%})")
                elif util.max_utilization < 0.30:
                    sel = auto_select(governing.max_bm, governing.max_sf, grade=grade)
                    st.info(f"Section is oversized (util < 30%). Lighter option: **{sel.recommended.name}** "
                           f"({sel.recommended.weight:.0f} kg/m², util = {sel.utilization.max_utilization:.1%})")

    # ── Diagrams ──
    with tab_plt:
        res_fe = st.session_state.get('cant_fe')
        res_bl = st.session_state.get('cant_bl')
        if not res_fe and not res_bl:
            st.info("Run the analysis first.")
        else:
            governing = res_fe or res_bl
            exc_d = st.session_state.excavation_depth

            # ── Combined Cross-Section ──
            try:
                from engine.section_library import get_section_by_name as _gsbn2
                _sec2 = _gsbn2(st.session_state.wall_section)
                _fy2 = 240 if '240' in st.session_state.get('wall_grade', 'S355GP') else 355
                _Md2 = _fy2 * _sec2.plastic_modulus * 1e-3 / 1.10 if _sec2 else 0
            except Exception:
                _Md2 = 0

            # Wrap cantilever result to match expected interface
            class _CantWrap:
                def __init__(self, r):
                    self.depths = r.depths
                    self.bending_moments = r.bending_moments
                    self.shear_forces = r.shear_forces
                    self.deflections = r.deflections
                    self.anchor_reactions = []
                    self.toe_reaction = getattr(r, 'toe_force', 0)

            fig_comb = plot_combined_crosssection(
                layers=st.session_state.layers,
                gwt_behind=st.session_state.gwt_behind,
                gwt_front=st.session_state.gwt_front,
                excavation_depth=exc_d,
                wall_toe=governing.total_wall_length,
                anchors=[],
                result=_CantWrap(governing),
                surcharge=st.session_state.surcharge,
                section_name=st.session_state.wall_section,
                Md=_Md2,
                title=f"{st.session_state.project_name} — Cantilever Wall",
            )
            st.plotly_chart(fig_comb, use_container_width=True)

            st.markdown("---")

            # ── Soil profile cross-section ──
            fig_soil = plot_soil_profile(
                layers=st.session_state.layers,
                gwt_behind=st.session_state.gwt_behind,
                gwt_front=st.session_state.gwt_front,
                excavation_depth=exc_d,
                total_wall_length=governing.total_wall_length,
                title="Geotechnical Cross-Section — Cantilever Wall",
            )
            st.plotly_chart(fig_soil, use_container_width=True)

            # ── BM / SF / Deflection ──
            traces = []
            if res_fe: traces.append(("Free Earth", res_fe, '#1A5276'))
            if res_bl: traces.append(("Blum", res_bl, '#922B21'))
            fig_forces = plot_internal_forces(
                results=traces,
                excavation_depth=exc_d,
                title="Internal Forces — Cantilever Wall",
            )
            st.plotly_chart(fig_forces, use_container_width=True)

            # ── Pressure distribution ──
            fig_press = plot_pressure_distribution(
                depths=governing.depths,
                active=governing.active_pressures,
                passive=governing.passive_pressures,
                excavation_depth=exc_d,
                gwt_behind=st.session_state.gwt_behind,
                pivot_depth=getattr(governing, 'pivot_depth', None),
                title="Active & Passive Pressure Distribution",
            )
            st.plotly_chart(fig_press, use_container_width=True)

            # ── Net pressure ──
            fig_net = plot_net_pressure(
                depths=governing.depths,
                net_pressures=governing.net_pressures,
                excavation_depth=exc_d,
                pivot_depth=getattr(governing, 'pivot_depth', None),
                title="Net Pressure Diagram",
            )
            st.plotly_chart(fig_net, use_container_width=True)

    # ── Design Table ──
    with tab_tbl:
        st.subheader("Quick Design Table — Cantilever Wall")
        st.caption("Free Earth Support method, your current soil profile, across excavation depths")
        exc_range = st.slider("Excavation depth range (m)", 2.0, 15.0, (2.0, 8.0), 1.0, key="dt_range")
        try:
            project = build_project()
            depths_list = list(range(int(exc_range[0]), int(exc_range[1]) + 1))
            table = cantilever_design_table(project, exc_depths=depths_list)
            tbl_rows = []
            for r in table:
                tbl_rows.append({
                    "Exc (m)": r['exc_depth'],
                    "Embedment (m)": f"{r['embedment']:.2f}",
                    "Total (m)": f"{r['total_length']:.2f}",
                    "Max BM (kN·m/m)": f"{r['max_bm']:.1f}",
                    "Max SF (kN/m)": f"{r['max_sf']:.1f}",
                    "FOS": f"{r['fos_moment']:.2f}",
                    "Status": r['status'],
                })
            st.dataframe(tbl_rows, use_container_width=True)

            # BM vs depth plot
            from plots import _layout_defaults
            fig_dt = make_subplots(rows=1, cols=2,
                subplot_titles=(
                    "<b>Max BM vs Excavation</b>",
                    "<b>Wall Length vs Excavation</b>",
                ))
            fig_dt.add_trace(go.Scatter(
                x=[r['exc_depth'] for r in table], y=[r['max_bm'] for r in table],
                mode='lines+markers+text', name='Max BM',
                text=[f"{r['max_bm']:.0f}" for r in table], textposition='top center',
                textfont=dict(size=10),
                line=dict(color='#1A5276', width=2.5),
                marker=dict(size=8, color='#1A5276')), row=1, col=1)
            fig_dt.add_trace(go.Scatter(
                x=[r['exc_depth'] for r in table], y=[r['total_length'] for r in table],
                mode='lines+markers+text', name='Total Length',
                text=[f"{r['total_length']:.1f}" for r in table], textposition='top center',
                textfont=dict(size=10),
                line=dict(color='#228B22', width=2.5),
                marker=dict(size=8, color='#228B22')), row=1, col=2)
            fig_dt.update_xaxes(title="<b>Excavation Depth (m)</b>", row=1, col=1)
            fig_dt.update_xaxes(title="<b>Excavation Depth (m)</b>", row=1, col=2)
            fig_dt.update_yaxes(title="<b>Max BM (kN·m/m)</b>", row=1, col=1)
            fig_dt.update_yaxes(title="<b>Wall Length (m)</b>", row=1, col=2)
            _layout_defaults(fig_dt, height=380)
            fig_dt.update_layout(showlegend=False)
            st.plotly_chart(fig_dt, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating table: {e}")

    with tab_rpt:
        res_fe = st.session_state.get('cant_fe')
        res_bl = st.session_state.get('cant_bl')
        if res_fe is None and res_bl is None:
            st.info("Run the analysis first, then generate the report.")
        else:
            if st.button("Generate Word Report", type="primary", key="cant_gen_report"):
                with st.spinner("Generating..."):
                    try:
                        from reports.report_generator import generate_cantilever_report
                        from engine.section_library import get_section_by_name
                        sec_obj = get_section_by_name(st.session_state.wall_section)
                        wall_toe_cant = st.session_state.excavation_depth * 2.5  # approximate
                        if res_fe and hasattr(res_fe, 'total_wall_length'):
                            wall_toe_cant = res_fe.total_wall_length
                        buf = generate_cantilever_report(
                            project=build_project(),
                            result_fe=res_fe, result_blum=res_bl,
                            wall_toe=wall_toe_cant,
                            EI=st.session_state.wall_EI,
                            wall_section_name=st.session_state.wall_section,
                            layers_ui=st.session_state.layers,
                            exc_depth=st.session_state.excavation_depth,
                            surcharge=st.session_state.surcharge,
                            gwt_behind=st.session_state.gwt_behind,
                            gwt_front=st.session_state.gwt_front,
                            section_obj=sec_obj,
                            project_name=st.session_state.project_name,
                            location=st.session_state.location,
                            firm_name=st.session_state.get('firm_name', ''),
                            engineer_name=st.session_state.get('engineer_name', ''),
                            revision=st.session_state.get('revision', 'R0'),
                        )
                        fname = f"{st.session_state.project_name.replace(' ','_')}_Cantilever_Report.docx"
                        st.download_button("Download Report", data=buf,
                            file_name=fname,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            type="primary")
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE: STAGED EXCAVATION
# ══════════════════════════════════════════════════════════════
elif st.session_state.analysis_mode == "Staged Excavation":
    st.markdown(f"## {st.session_state.project_name} — Staged Excavation Analysis")
    st.caption(_page_caption())

    tab_setup, tab_results, tab_diagrams, tab_rpt_staged = st.tabs(
        ["Setup & Anchors", "Results", "Diagrams", "Report"])

    # ── Setup ──
    with tab_setup:
        c1, c2 = st.columns(2)
        with c1:
            wall_toe_staged = st.number_input("Wall Toe (m below GL)",
                min_value=st.session_state.excavation_depth + 0.5,
                max_value=50.0,
                value=max(st.session_state.get('wall_toe', st.session_state.excavation_depth + 2.0),
                          st.session_state.excavation_depth + 0.5),
                step=0.5, key="staged_wall_toe")
        with c2:
            # Section picker
            all_secs = get_all_sections()
            sec_names = [s.name for s in all_secs]
            staged_sec_idx = st.selectbox("Sheet Pile Section", range(len(sec_names)),
                format_func=lambda i: sec_names[i], key="staged_sec_sel")
            staged_sec = all_secs[staged_sec_idx]
            staged_EI = staged_sec.EI_per_m
            st.caption(f"EI = {staged_EI:.0f} kN-m2/m | {staged_sec.manufacturer}")

        c3, c4, c5 = st.columns(3)
        with c3:
            exc_step = st.number_input("Excavation Step (m)", 0.5, 3.0, 1.0, 0.5, key="staged_exc_step")
        with c4:
            working_margin = st.number_input("Working Margin (m)", 0.2, 2.0, 0.5, 0.1, key="staged_margin")
        with c5:
            staged_fos_p = st.number_input("FOS on Passive", 1.0, 3.0, 1.5, 0.1, key="staged_fos_p")

        st.markdown("#### Anchor Layout")
        n_anchors_staged = st.number_input("Number of Anchors", 0, 10, 2, key="staged_n_anchors")

        staged_anchors_def = []
        for i in range(n_anchors_staged):
            st.markdown(f"**Anchor {i+1}**")
            ac1, ac2, ac3, ac4 = st.columns(4)
            a_level = ac1.number_input(f"Level (m)", 0.5, 30.0,
                value=min(1.5 + i * 2.0, st.session_state.excavation_depth - 0.5),
                step=0.5, key=f"stg_a_lvl_{i}")
            a_type = ac2.selectbox(f"Type", ["rebar", "sda", "prestressed"], key=f"stg_a_type_{i}")
            a_incl = ac3.number_input(f"Incl (deg)", 0.0, 45.0, 20.0, 5.0, key=f"stg_a_incl_{i}")
            a_spc = ac4.number_input(f"Spacing (m)", 1.0, 6.0, 3.0, 0.5, key=f"stg_a_spc_{i}")
            staged_anchors_def.append({
                'level': a_level, 'type': a_type, 'incl': a_incl, 'spc': a_spc
            })

        # Preview auto-generated stages
        if st.checkbox("Preview construction stages", key="staged_preview"):
            from engine.anchored_wall import Anchor as AnchorCls
            preview_anchors = [
                AnchorCls(level=a['level'], anchor_type=a['type'],
                          inclination=a['incl'], horizontal_spacing=a['spc'])
                for a in staged_anchors_def
            ]
            preview_stages = generate_stages(
                excavation_depth=st.session_state.excavation_depth,
                anchors=preview_anchors, exc_step=exc_step,
                working_margin=working_margin,
            )
            import pandas as pd
            st.dataframe(pd.DataFrame([
                {'Stage': s.stage_number, 'Description': s.description,
                 'Exc (m)': s.excavation_depth, 'Active Anchors': len(s.active_anchor_indices)}
                for s in preview_stages
            ]), use_container_width=True, hide_index=True)

        # Run button
        if st.button("🚀 Run Staged Analysis", type="primary", use_container_width=True, key="run_staged"):
            from engine.anchored_wall import Anchor as AnchorCls
            anchors_obj = [
                AnchorCls(level=a['level'], anchor_type=a['type'],
                          inclination=a['incl'], horizontal_spacing=a['spc'])
                for a in staged_anchors_def
            ]
            project = build_project()
            try:
                staged_res = analyze_staged_excavation(
                    project=project,
                    anchors=anchors_obj,
                    wall_toe_level=wall_toe_staged,
                    EI=staged_EI,
                    exc_step=exc_step,
                    working_margin=working_margin,
                    fos_passive=staged_fos_p,
                )
                st.session_state.staged_result = staged_res
                st.session_state.staged_anchors = staged_anchors_def
                st.session_state.staged_sec = staged_sec
                st.success(f"Analysis complete! {staged_res.n_stages} stages analyzed.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # ── Results ──
    with tab_results:
        staged_res = st.session_state.get('staged_result')
        if staged_res is None:
            st.info("Run the analysis first.")
        else:
            # Design values
            st.markdown("### Design Values (Envelope)")
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Design BM", f"{staged_res.design_bm:.1f} kN·m/m",
                       f"Stage {staged_res.design_bm_stage}")
            dc2.metric("Design SF", f"{staged_res.design_sf:.1f} kN/m",
                       f"Stage {staged_res.design_sf_stage}")
            dc3.metric("Design Defl", f"{staged_res.design_defl:.1f} mm",
                       f"Stage {staged_res.design_defl_stage}")
            dc4.metric("Stages", str(staged_res.n_stages))

            # Section adequacy
            sec = st.session_state.get('staged_sec')
            if sec:
                grade = SteelGrade.S355GP
                util_res = check_section(sec, grade, staged_res.design_bm, staged_res.design_sf)
                icon = "✅" if util_res.utilization_combined <= 1.0 else "❌"
                st.markdown(f"**{sec.name}** (S355GP): {util_res.utilization_combined:.1%} utilization — {icon}")

            # Stage summary table
            st.markdown("### Per-Stage Summary")
            import pandas as pd
            df = pd.DataFrame(staged_res.summary)
            df.columns = ['Stage', 'Description', 'Exc (m)', 'Anchors',
                          'Max BM', 'Max SF', 'Max Defl', 'Status']

            # Highlight governing stages
            def highlight_governing(row):
                styles = [''] * len(row)
                if abs(row['Max BM'] - staged_res.design_bm) < 0.5:
                    styles[4] = 'background-color: #FADBD8; font-weight: bold'
                if abs(row['Max SF'] - staged_res.design_sf) < 0.5:
                    styles[5] = 'background-color: #FADBD8; font-weight: bold'
                if abs(row['Max Defl'] - staged_res.design_defl) < 0.5:
                    styles[6] = 'background-color: #FADBD8; font-weight: bold'
                return styles

            st.dataframe(
                df.style.apply(highlight_governing, axis=1).format({
                    'Exc (m)': '{:.1f}', 'Max BM': '{:.1f}',
                    'Max SF': '{:.1f}', 'Max Defl': '{:.1f}',
                }),
                use_container_width=True, hide_index=True,
            )

            # Per-stage bar chart
            fig_bars = plot_stage_summary_bars(staged_res)
            st.plotly_chart(fig_bars, use_container_width=True)

    # ── Diagrams ──
    with tab_diagrams:
        staged_res = st.session_state.get('staged_result')
        if staged_res is None:
            st.info("Run the analysis first.")
        else:
            exc_d = st.session_state.excavation_depth
            wall_toe_d = st.session_state.get('staged_wall_toe', exc_d + 2)

            # ── Combined Cross-Section (final stage) ──
            try:
                from engine.section_library import get_section_by_name as _gsbn3
                _sec3 = _gsbn3(st.session_state.get('staged_section', st.session_state.wall_section))
                _fy3 = 240 if '240' in st.session_state.get('wall_grade', 'S355GP') else 355
                _Md3 = _fy3 * _sec3.plastic_modulus * 1e-3 / 1.10 if _sec3 else 0
            except Exception:
                _Md3 = 0

            # Use final stage for combined plot
            final_stage = staged_res.stages[-1]
            class _StagedWrap:
                def __init__(self, sr):
                    self.depths = sr.depths
                    self.bending_moments = sr.bending_moments
                    self.shear_forces = sr.shear_forces
                    self.deflections = sr.deflections
                    self.anchor_reactions = sr.anchor_reactions
                    self.toe_reaction = sr.toe_reaction

            fig_comb_s = plot_combined_crosssection(
                layers=st.session_state.layers,
                gwt_behind=st.session_state.gwt_behind,
                gwt_front=st.session_state.gwt_front,
                excavation_depth=exc_d,
                wall_toe=wall_toe_d,
                anchors=st.session_state.get('staged_anchors', st.session_state.anchors),
                result=_StagedWrap(final_stage),
                surcharge=st.session_state.surcharge,
                section_name=st.session_state.get('staged_section', st.session_state.wall_section),
                Md=_Md3,
                title=f"{st.session_state.project_name} — Final Stage ({final_stage.stage.description})",
            )
            st.plotly_chart(fig_comb_s, use_container_width=True)

            st.markdown("---")

            # Soil profile
            fig_soil = plot_soil_profile(
                layers=st.session_state.layers,
                gwt_behind=st.session_state.gwt_behind,
                gwt_front=st.session_state.gwt_front,
                excavation_depth=exc_d,
                wall_toe=wall_toe_d,
                anchors=st.session_state.get('staged_anchors', []),
                title="Geotechnical Cross-Section",
            )
            st.plotly_chart(fig_soil, use_container_width=True)

            # Envelope plot
            fig_env = plot_staged_envelope(
                staged_result=staged_res,
                excavation_depth=exc_d,
                anchors=st.session_state.get('staged_anchors', []),
            )
            st.plotly_chart(fig_env, use_container_width=True)

            # Individual stage selector
            st.markdown("### Individual Stage Diagrams")
            stage_options = [f"Stage {s.stage.stage_number}: {s.stage.description}"
                            for s in staged_res.stages if s.max_bm > 0.01]
            if stage_options:
                sel_stage = st.selectbox("Select stage", stage_options, key="staged_sel_stage")
                sel_idx = int(sel_stage.split(":")[0].replace("Stage ", ""))
                sel_sr = [s for s in staged_res.stages if s.stage.stage_number == sel_idx][0]

                fig_ind = plot_internal_forces(
                    results=[(f"Stage {sel_idx}", sel_sr, '#1A5276')],
                    excavation_depth=sel_sr.stage.excavation_depth,
                    title=f"Stage {sel_idx}: {sel_sr.stage.description}",
                )
                st.plotly_chart(fig_ind, use_container_width=True)

    with tab_rpt_staged:
        staged_res = st.session_state.get('staged_result')
        if staged_res is None:
            st.info("Run the staged analysis first, then generate the report.")
        else:
            if st.button("Generate Word Report", type="primary", key="staged_gen_report"):
                with st.spinner("Generating..."):
                    try:
                        from reports.report_generator import generate_staged_report
                        from engine.section_library import get_section_by_name
                        sec_obj = get_section_by_name(st.session_state.get('staged_section', st.session_state.wall_section))
                        buf = generate_staged_report(
                            project=build_project(),
                            staged_result=staged_res,
                            anchors=st.session_state.anchors,
                            wall_toe=st.session_state.get('staged_toe', st.session_state.wall_toe),
                            EI=st.session_state.wall_EI,
                            wall_section_name=st.session_state.get('staged_section', st.session_state.wall_section),
                            layers_ui=st.session_state.layers,
                            anchors_ui=st.session_state.anchors,
                            exc_depth=st.session_state.excavation_depth,
                            surcharge=st.session_state.surcharge,
                            gwt_behind=st.session_state.gwt_behind,
                            gwt_front=st.session_state.gwt_front,
                            section_obj=sec_obj,
                            project_name=st.session_state.project_name,
                            location=st.session_state.location,
                            firm_name=st.session_state.get('firm_name', ''),
                            engineer_name=st.session_state.get('engineer_name', ''),
                            revision=st.session_state.get('revision', 'R0'),
                        )
                        fname = f"{st.session_state.project_name.replace(' ','_')}_Staged_Report.docx"
                        st.download_button("Download Report", data=buf,
                            file_name=fname,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            type="primary")
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")
# ══════════════════════════════════════════════════════════════
# PAGE: ANCHORED WALL (Module 5)
# ══════════════════════════════════════════════════════════════
elif st.session_state.analysis_mode == "Anchored Wall":
    st.markdown(f"## {st.session_state.project_name} — Anchored Wall Analysis")
    st.caption(_page_caption(f"Wall toe: {st.session_state.wall_toe}m"))

    tab_input, tab_results, tab_plots, tab_report = st.tabs([
        "Anchors", "Results", "Diagrams", "Report"
    ])

    # ── Anchor Config ──
    with tab_input:
        col_anc, col_extra = st.columns([3, 2])

        with col_anc:
            st.markdown("### Wall & Anchors")
            # Section from library
            all_secs = get_all_sections()
            sec_names = [s.name for s in all_secs]
            default_idx = sec_names.index(st.session_state.wall_section) if st.session_state.wall_section in sec_names else 0
            sel_sec = st.selectbox("Sheet Pile Section", sec_names, index=default_idx, key="anc_sec")
            sec_obj = get_section_by_name(sel_sec)
            if sec_obj:
                st.session_state.wall_section = sel_sec
                st.session_state.wall_EI = sec_obj.EI_per_m
                st.caption(f"EI = {sec_obj.EI_per_m:.0f} kN·m²/m | Zp = {sec_obj.plastic_modulus} cm³/m | "
                          f"{sec_obj.manufacturer}")

            st.session_state.wall_toe = st.number_input(
                "Wall toe depth (m)", 1.0, 30.0, st.session_state.wall_toe, 0.1, key="anc_toe")

            n_anc = st.number_input("Anchor levels", 1, 6, len(st.session_state.anchors), 1, key="anc_n")
            while len(st.session_state.anchors) < n_anc:
                prev = st.session_state.anchors[-1].copy()
                prev['level'] = prev['level'] + 1.5
                st.session_state.anchors.append(prev)
            while len(st.session_state.anchors) > n_anc:
                st.session_state.anchors.pop()

            for i, a in enumerate(st.session_state.anchors):
                st.markdown(f"---\n**Anchor {i+1}**")
                c1, c2, c3, c4 = st.columns(4)
                a['level'] = c1.number_input(f"Level##A{i}", 0.5, 30.0, a['level'], 0.5, key=f"al_{i}")
                a['type'] = c2.selectbox(f"Type##A{i}", ['rebar', 'sda', 'prestressed'],
                    index=['rebar', 'sda', 'prestressed'].index(a['type']), key=f"at_{i}")
                a['incl'] = c3.number_input(f"Incl°##A{i}", 0.0, 45.0, a['incl'], 5.0, key=f"ai_{i}")
                a['spacing'] = c4.number_input(f"Spc (m)##A{i}", 1.0, 6.0, a['spacing'], 0.5, key=f"as_{i}")
                c5, c6 = st.columns(2)
                a['bond_stress'] = c5.number_input(
                    f"Bond (kPa)##A{i}", 50.0, 1000.0, a['bond_stress'], 25.0, key=f"ab_{i}")
                a['drill_dia'] = c6.number_input(
                    f"Drill mm##A{i}", 50.0, 250.0, a['drill_dia'], 5.0, key=f"ad_{i}")
                if a['type'] == 'rebar':
                    cr1, cr2, cr3 = st.columns(3)
                    a['rebar_dia'] = cr1.selectbox(f"Bar mm##A{i}", [12,16,20,25,28,32,36,40],
                        index=[12,16,20,25,28,32,36,40].index(int(a['rebar_dia'])), key=f"ard_{i}")
                    a['rebar_fy'] = cr2.selectbox(f"fy##A{i}", [415,500,550],
                        index=[415,500,550].index(int(a['rebar_fy'])), key=f"arf_{i}")
                    a['rebar_count'] = cr3.number_input(f"Bars##A{i}", 1, 4, a['rebar_count'], 1, key=f"arc_{i}")

        with col_extra:
            st.markdown("### Additional")
            st.markdown("**🌬️ Wind Barrier (IS 875-3)**")
            st.session_state.wind_barrier = st.toggle("Include wind", st.session_state.wind_barrier, key="anc_wind")
            if st.session_state.wind_barrier:
                st.session_state.barrier_height = st.number_input(
                    "Height (m)", 2.0, 15.0, st.session_state.barrier_height, 1.0, key="anc_bh")
                st.session_state.Vb = st.number_input(
                    "Vb (m/s)", 30.0, 60.0, st.session_state.Vb, 1.0, key="anc_vb")
                F, M, pz, Vz, k2 = compute_wind_load(
                    st.session_state.barrier_height, st.session_state.Vb, st.session_state.terrain_cat)
                st.info(f"F = {F:.1f} kN/m | M = {M:.1f} kN·m/m")

            st.markdown("---\n**🔧 Spring Model**")
            st.session_state.spring_model = st.toggle("Springs", st.session_state.spring_model, key="anc_spr")
            if st.session_state.spring_model:
                st.session_state.anchor_k = st.number_input(
                    "k (kN/mm)", 0.1, 100.0, st.session_state.anchor_k, 0.5, key="anc_k")

        st.markdown("---")
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True, key="anc_run"):
            try:
                project = build_project()
                anchors = build_anchors()
                result = analyze_anchored_wall(
                    project, anchors, st.session_state.wall_toe,
                    st.session_state.wall_EI, 100)
                st.session_state.result = result

                if st.session_state.wind_barrier:
                    F_w, M_w, _, _, _ = compute_wind_load(
                        st.session_state.barrier_height, st.session_state.Vb, st.session_state.terrain_cat)
                    st.session_state.result_wind = analyze_anchored_wall(
                        project, anchors, st.session_state.wall_toe,
                        st.session_state.wall_EI, 100, point_loads=[(0.0, F_w, M_w)])
                else:
                    st.session_state.result_wind = None
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # ── Results ──
    with tab_results:
        result = st.session_state.get('result')
        if result is None:
            st.info("Run the analysis first.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max BM", f"{result.max_moment:.1f} kN·m/m")
            m2.metric("Max SF", f"{result.max_shear:.1f} kN/m")
            total_rxn = sum(result.anchor_reactions) + result.toe_reaction
            m3.metric("Total Reaction", f"{total_rxn:.1f} kN/m")
            m4.metric("Max Defl", f"{result.max_deflection:.2f} mm")

            # Section utilization
            sec_obj = get_section_by_name(st.session_state.wall_section)
            if sec_obj:
                gov_m = result.max_moment
                gov_v = result.max_shear
                rw = st.session_state.get('result_wind')
                if rw:
                    gov_m = max(gov_m, rw.max_moment)
                    gov_v = max(gov_v, rw.max_shear)
                util = check_section(sec_obj, SteelGrade.S355GP, gov_m, gov_v)
                icon = '✅' if util.overall_status == 'OK' else '❌'
                st.markdown(f"**{sec_obj.name} (S355GP): {util.max_utilization:.1%} utilization — {icon}**")

            # Anchor reactions table
            st.markdown("#### Reactions")
            rxn_data = []
            for i, rxn in enumerate(result.anchor_reactions):
                a = st.session_state.anchors[i]
                rxn_data.append({
                    "Anchor": f"#{i+1} at {a['level']}m",
                    "Reaction (kN/m)": f"{rxn:.1f}",
                    "Per anchor (kN)": f"{rxn * a['spacing']:.1f}",
                })
            rxn_data.append({"Anchor": "Toe", "Reaction (kN/m)": f"{result.toe_reaction:.1f}", "Per anchor (kN)": "—"})
            st.dataframe(rxn_data, use_container_width=True)

            # Anchor design
            if result.anchor_designs:
                st.markdown("#### Anchor Design Checks")
                for idx_ad, ad in enumerate(result.anchor_designs):
                    atype = st.session_state.anchors[idx_ad].get('type', 'rebar') if idx_ad < len(st.session_state.anchors) else 'rebar'
                    is_rebar = atype == 'rebar'
                    is_sda = atype == 'sda'
                    icon = '✅' if ad.status == 'OK' else '⚠️'
                    with st.expander(f"{icon} {ad.label} — {ad.status}", expanded=True):
                        dc1, dc2, dc3 = st.columns(3)
                        flbl = "Bar Force" if is_rebar else ("Anchor Force" if is_sda else "Tendon Force")
                        foslbl = "FOS (Bar)" if is_rebar else ("FOS (Anchor)" if is_sda else "FOS (Tendon)")
                        dc1.metric(flbl, f"{ad.tendon_force:.1f} kN")
                        dc2.metric("FOS (Bond)", f"{ad.fos_bond_actual:.2f}",
                                  "OK" if ad.fos_bond_actual >= 2.0 else "LOW")
                        dc3.metric(foslbl, f"{ad.fos_tendon_actual:.2f}",
                                  "OK" if ad.fos_tendon_actual >= 1.5 else "LOW")

    # ── Plots ──
    with tab_plots:
        result = st.session_state.get('result')
        if result is None:
            st.info("Run the analysis first.")
        else:
            result_wind = st.session_state.get('result_wind')
            exc_d = st.session_state.excavation_depth

            # ── Combined Cross-Section (DeepEX-style) ──
            try:
                from engine.section_library import get_section_by_name as _gsbn
                _sec = _gsbn(st.session_state.wall_section)
                _fy = 240 if '240' in st.session_state.get('wall_grade', 'S355GP') else 355
                _Md = _fy * _sec.plastic_modulus * 1e-3 / 1.10 if _sec else 0
            except Exception:
                _Md = 0

            fig_combined = plot_combined_crosssection(
                layers=st.session_state.layers,
                gwt_behind=st.session_state.gwt_behind,
                gwt_front=st.session_state.gwt_front,
                excavation_depth=exc_d,
                wall_toe=st.session_state.wall_toe,
                anchors=st.session_state.anchors,
                result=result, result_wind=result_wind,
                surcharge=st.session_state.surcharge,
                section_name=st.session_state.wall_section,
                Md=_Md,
                title=f"{st.session_state.project_name} — Combined Analysis",
            )
            st.plotly_chart(fig_combined, use_container_width=True)

            st.markdown("---")

            # ── Soil profile cross-section ──
            fig_soil = plot_soil_profile(
                layers=st.session_state.layers,
                gwt_behind=st.session_state.gwt_behind,
                gwt_front=st.session_state.gwt_front,
                excavation_depth=exc_d,
                wall_toe=st.session_state.wall_toe,
                anchors=st.session_state.anchors,
                title="Geotechnical Cross-Section — Anchored Wall",
            )
            st.plotly_chart(fig_soil, use_container_width=True)

            # ── BM / SF / Deflection ──
            traces = [("Earth only", result, '#1A5276')]
            if result_wind:
                traces.append(("Earth + Wind", result_wind, '#922B21'))

            # Adapt result object for plot helper (anchored wall uses different attr names)
            class _Wrap:
                def __init__(self, r):
                    self.depths = r.depths
                    self.bending_moments = r.bending_moments
                    self.shear_forces = r.shear_forces
                    self.deflections = r.deflections

            wrapped = [(_l, _Wrap(_r), _c) for _l, _r, _c in traces]
            fig_forces = plot_internal_forces(
                results=wrapped,
                excavation_depth=exc_d,
                anchors=st.session_state.anchors,
                title="Internal Forces — Anchored Wall",
            )
            st.plotly_chart(fig_forces, use_container_width=True)

    # ── Report ──
    with tab_report:
        result = st.session_state.get('result')
        if result is None:
            st.info("Run the analysis first, then generate the report.")
        else:
            st.subheader("Generate Report")
            if st.button("Generate Word Report", type="primary", key="gen_report"):
                with st.spinner("Generating..."):
                    try:
                        from reports.report_generator import generate_anchored_wall_report
                        from engine.section_library import get_section_by_name
                        sec_obj = get_section_by_name(st.session_state.wall_section)
                        buf = generate_anchored_wall_report(
                            project=build_project(), anchors=build_anchors(),
                            result=result,
                            result_wind=st.session_state.get('result_wind'),
                            wall_toe=st.session_state.wall_toe,
                            EI=st.session_state.wall_EI,
                            wall_section_name=st.session_state.wall_section,
                            layers_ui=st.session_state.layers,
                            anchors_ui=st.session_state.anchors,
                            exc_depth=st.session_state.excavation_depth,
                            surcharge=st.session_state.surcharge,
                            gwt_behind=st.session_state.gwt_behind,
                            gwt_front=st.session_state.gwt_front,
                            section_obj=sec_obj,
                            project_name=st.session_state.project_name,
                            location=st.session_state.location,
                            firm_name=st.session_state.get('firm_name', ''),
                            engineer_name=st.session_state.get('engineer_name', ''),
                            revision=st.session_state.get('revision', 'R0'),
                            wind_barrier=st.session_state.get('wind_barrier', False),
                            barrier_height=st.session_state.get('barrier_height', 0),
                            Vb=st.session_state.get('Vb', 44),
                        )
                        fname = f"{st.session_state.project_name.replace(' ','_')}_Anchored_Wall_Report.docx"
                        st.download_button("Download Report", data=buf,
                            file_name=fname,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            type="primary")
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")
