"""
Production-quality geotechnical plots.
Konstantinokas / PLAXIS-style diagrams for deep excavation analysis.

Features:
- Soil stratigraphy with hatching patterns and color coding
- Wall cross-section with anchor positions
- Annotated BM/SF/Deflection diagrams with peak values
- Pressure distribution with net pressure shading
- GWT indicator
- Professional typography and color palette


"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Optional, Tuple

# ══════════════════════════════════════════════════════════════
# COLOR PALETTE (professional geotechnical)
# ══════════════════════════════════════════════════════════════

SOIL_COLORS = {
    'Fill':       {'fill': '#D4A574', 'line': '#8B6914', 'pattern': '/', 'label': 'Made Ground / Fill'},
    'Sand':       {'fill': '#F5DEB3', 'line': '#DAA520', 'pattern': '.', 'label': 'Sand'},
    'Clay':       {'fill': '#A0826D', 'line': '#654321', 'pattern': '-', 'label': 'Clay'},
    'Silt':       {'fill': '#C4A882', 'line': '#8B7355', 'pattern': '|', 'label': 'Silt'},
    'Gravel':     {'fill': '#D2B48C', 'line': '#A0522D', 'pattern': 'o', 'label': 'Gravel'},
    'Rock':       {'fill': '#B0B0B0', 'line': '#505050', 'pattern': 'x', 'label': 'Rock'},
    'CWR':        {'fill': '#9B9B7A', 'line': '#6B6B4F', 'pattern': '+', 'label': 'Weathered Rock'},
    'Residual':   {'fill': '#BC9F77', 'line': '#8B7355', 'pattern': '\\', 'label': 'Residual Soil'},
    'default':    {'fill': '#C8B896', 'line': '#7A6B5A', 'pattern': '', 'label': 'Soil'},
}

WALL_COLOR = '#2E4057'
ANCHOR_COLOR = '#B22222'
GWT_COLOR = '#1E90FF'
EXCAVATION_COLOR = '#FFFFFF'
ACTIVE_COLOR = '#8B4513'
PASSIVE_COLOR = '#228B22'
BM_COLOR = '#1A5276'
SF_COLOR = '#922B21'
DEFL_COLOR = '#6C3483'
NET_COLOR = '#762A83'
GRID_COLOR = '#E5E5E5'

# Professional font settings
FONT_FAMILY = "Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
TITLE_SIZE = 20
AXIS_SIZE = 16
TICK_SIZE = 15
ANNOT_SIZE = 14


def _classify_soil(name: str) -> str:
    """Map soil layer name to a color category."""
    n = name.upper()
    if 'FILL' in n or 'MADE' in n:
        return 'Fill'
    if 'SAND' in n:
        return 'Sand'
    if 'CLAY' in n:
        return 'Clay'
    if 'SILT' in n:
        return 'Silt'
    if 'GRAVEL' in n:
        return 'Gravel'
    if 'ROCK' in n or 'BASALT' in n or 'GRANITE' in n:
        return 'Rock'
    if 'CWR' in n or 'BRECCIA' in n or 'WEATHER' in n:
        return 'CWR'
    if 'RESIDUAL' in n or 'LATERITE' in n:
        return 'Residual'
    return 'default'


def _layout_defaults(fig, title="", height=550):
    """Apply consistent professional styling to a figure."""
    fig.update_layout(
        height=height,
        template="plotly_white",
        title=dict(text=title, font=dict(size=TITLE_SIZE, family=FONT_FAMILY, color='#1C2833')),
        font=dict(family=FONT_FAMILY, size=TICK_SIZE, color='#2C3E50'),
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='white',
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(
            font=dict(size=ANNOT_SIZE),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#D5D8DC',
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR, gridwidth=0.5,
        zeroline=True, zerolinecolor='#AAB7B8', zerolinewidth=0.8,
        tickfont=dict(size=TICK_SIZE),
        title_font=dict(size=AXIS_SIZE),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR, gridwidth=0.5,
        zeroline=True, zerolinecolor='#AAB7B8', zerolinewidth=0.8,
        tickfont=dict(size=TICK_SIZE),
        title_font=dict(size=AXIS_SIZE),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# SOIL STRATIGRAPHY PLOT
# ══════════════════════════════════════════════════════════════

def plot_soil_profile(
    layers: List[Dict],
    gwt_behind: float,
    gwt_front: float,
    excavation_depth: float,
    wall_toe: Optional[float] = None,
    anchors: Optional[List[Dict]] = None,
    total_wall_length: Optional[float] = None,
    title: str = "Geotechnical Cross-Section",
) -> go.Figure:
    """
    Konstantinokas-style soil profile with wall, anchors, and GWT.

    Parameters:
        layers: list of {'name', 'top', 'bottom', 'gamma', 'phi', 'c'}
        gwt_behind, gwt_front: GWT depths
        excavation_depth: m below GL
        wall_toe: wall toe depth (for anchored wall)
        anchors: list of {'level', 'incl', ...}
        total_wall_length: total wall length (for cantilever)
    """
    fig = go.Figure()

    # Determine extents
    max_depth = max(l['bottom'] for l in layers)
    if total_wall_length:
        max_depth = max(max_depth, total_wall_length + 1)
    if wall_toe:
        max_depth = max(max_depth, wall_toe + 1)

    x_left = -8   # behind wall (retained side)
    x_right = 8   # in front (excavation side)
    wall_x = 0    # wall position

    # ── Soil layers (behind wall — full depth) ──
    for lay in layers:
        cat = _classify_soil(lay['name'])
        col = SOIL_COLORS.get(cat, SOIL_COLORS['default'])

        # Behind wall (full depth)
        fig.add_trace(go.Scatter(
            x=[x_left, wall_x - 0.15, wall_x - 0.15, x_left, x_left],
            y=[lay['top'], lay['top'], lay['bottom'], lay['bottom'], lay['top']],
            fill='toself', fillcolor=col['fill'],
            line=dict(color=col['line'], width=1),
            name=lay['name'], showlegend=True,
            hovertemplate=f"<b>{lay['name']}</b><br>"
                         f"Depth: {lay['top']:.1f} – {lay['bottom']:.1f}m<br>"
                         f"γ = {lay['gamma']} kN/m³ | φ = {lay['phi']}° | c = {lay['c']} kPa"
                         f"<extra></extra>",
        ))

        # In front of wall (below excavation only)
        if lay['bottom'] > excavation_depth:
            top_front = max(lay['top'], excavation_depth)
            fig.add_trace(go.Scatter(
                x=[wall_x + 0.15, x_right, x_right, wall_x + 0.15, wall_x + 0.15],
                y=[top_front, top_front, lay['bottom'], lay['bottom'], top_front],
                fill='toself', fillcolor=col['fill'],
                line=dict(color=col['line'], width=1),
                showlegend=False,
            ))

    # ── Excavation void ──
    fig.add_trace(go.Scatter(
        x=[wall_x + 0.15, x_right, x_right, wall_x + 0.15, wall_x + 0.15],
        y=[0, 0, excavation_depth, excavation_depth, 0],
        fill='toself', fillcolor=EXCAVATION_COLOR,
        line=dict(color='#CCCCCC', width=1, dash='dot'),
        name='Excavation', showlegend=True,
    ))
    # Excavation level label
    fig.add_annotation(
        x=x_right - 0.5, y=excavation_depth,
        text=f"Exc. Level<br>{excavation_depth:.1f}m",
        showarrow=False, font=dict(size=ANNOT_SIZE, color='#922B21', family=FONT_FAMILY),
        xanchor='right', yanchor='top',
        bgcolor='rgba(255,255,255,0.85)', bordercolor='#922B21', borderwidth=1, borderpad=3,
    )

    # ── Wall ──
    toe = wall_toe or total_wall_length or excavation_depth
    fig.add_trace(go.Scatter(
        x=[wall_x - 0.12, wall_x + 0.12, wall_x + 0.12, wall_x - 0.12, wall_x - 0.12],
        y=[0, 0, toe, toe, 0],
        fill='toself', fillcolor=WALL_COLOR,
        line=dict(color='#1B2631', width=2),
        name=f'Wall (L={toe:.1f}m)', showlegend=True,
    ))

    # ── Anchors ──
    if anchors:
        for i, a in enumerate(anchors):
            level = a['level']
            incl_rad = np.radians(a.get('incl', 20))
            anc_len = 4.0  # display length
            x_end = wall_x - anc_len * np.cos(incl_rad)
            y_end = level + anc_len * np.sin(incl_rad)

            # Anchor line
            fig.add_trace(go.Scatter(
                x=[wall_x, x_end], y=[level, y_end],
                mode='lines+markers',
                line=dict(color=ANCHOR_COLOR, width=2.5),
                marker=dict(symbol=['circle', 'triangle-left'], size=[6, 10], color=ANCHOR_COLOR),
                name=f'Anchor {i+1} ({level}m)',
                showlegend=True,
            ))
            # Label
            fig.add_annotation(
                x=x_end - 0.3, y=y_end,
                text=f"A{i+1}: {level:.1f}m",
                showarrow=False, font=dict(size=13, color=ANCHOR_COLOR, family=FONT_FAMILY),
                xanchor='right',
            )

    # ── GWT lines ──
    # Behind wall
    fig.add_trace(go.Scatter(
        x=[x_left, wall_x - 0.15], y=[gwt_behind, gwt_behind],
        mode='lines', line=dict(color=GWT_COLOR, width=2, dash='dash'),
        name=f'GWT behind ({gwt_behind}m)', showlegend=True,
    ))
    # Triangle markers for GWT
    for xx in np.linspace(x_left + 1, wall_x - 1, 5):
        fig.add_annotation(
            x=xx, y=gwt_behind, text="▽", showarrow=False,
            font=dict(size=13, color=GWT_COLOR), yshift=-8,
        )

    # In front (if below excavation)
    if gwt_front < max_depth:
        fig.add_trace(go.Scatter(
            x=[wall_x + 0.15, x_right], y=[gwt_front, gwt_front],
            mode='lines', line=dict(color=GWT_COLOR, width=2, dash='dash'),
            name=f'GWT front ({gwt_front}m)', showlegend=True,
        ))

    # ── Ground level line ──
    fig.add_trace(go.Scatter(
        x=[x_left, x_right], y=[0, 0],
        mode='lines', line=dict(color='#1B4F08', width=3),
        showlegend=False,
    ))
    fig.add_annotation(x=x_left + 0.5, y=0, text="GL ±0.00",
                       showarrow=False, font=dict(size=ANNOT_SIZE, color='#1B4F08'),
                       yshift=-12)

    # ── Layer boundary labels ──
    for lay in layers:
        if lay['top'] > 0:
            fig.add_annotation(
                x=x_left + 0.3, y=lay['top'],
                text=f"─ {lay['top']:.1f}m", showarrow=False,
                font=dict(size=13, color='#5D6D7E', family=FONT_FAMILY),
                xanchor='left', yanchor='bottom',
            )

    # ── Depth scale (right side) ──
    for d in range(1, int(max_depth) + 1):
        fig.add_annotation(
            x=x_right - 0.2, y=d,
            text=f"{d}m", showarrow=False,
            font=dict(size=12, color='#ABB2B9'), xanchor='right',
        )

    # ── Layout ──
    fig.update_yaxes(autorange="reversed", title="Depth (m)", range=[-0.5, max_depth + 0.5],
                     showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    fig.update_xaxes(title="", showticklabels=False, range=[x_left - 0.5, x_right + 0.5],
                     showgrid=False)
    _layout_defaults(fig, title=title, height=550)
    fig.update_layout(
        legend=dict(orientation="v", y=0.5, x=1.02, xanchor='left',
                    font=dict(size=12)),
        margin=dict(l=60, r=180, t=60, b=40),
    )

    return fig


# ══════════════════════════════════════════════════════════════
# ANNOTATED BM / SF / DEFLECTION DIAGRAMS
# ══════════════════════════════════════════════════════════════

def plot_internal_forces(
    results: List[Tuple[str, object, str]],
    excavation_depth: float,
    anchors: Optional[List[Dict]] = None,
    title: str = "Internal Forces & Deflection",
) -> go.Figure:
    """
    Professional 3-panel BM/SF/Deflection plot with annotations.

    Parameters:
        results: list of (label, result_object, color) — result must have
                 .depths, .bending_moments/.bm, .shear_forces/.sf, .deflections
        excavation_depth: m
        anchors: list of {'level': float} for marker lines
        title: figure title
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "<b>Bending Moment</b><br><sup>kN·m/m</sup>",
            "<b>Shear Force</b><br><sup>kN/m</sup>",
            "<b>Deflection</b><br><sup>mm</sup>",
        ),
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    for label, res, color in results:
        # Get data (handle both naming conventions)
        depths = res.depths if hasattr(res, 'depths') else []
        bm = res.bending_moments if hasattr(res, 'bending_moments') else getattr(res, 'bm', [])
        sf = res.shear_forces if hasattr(res, 'shear_forces') else getattr(res, 'sf', [])
        defl = res.deflections if hasattr(res, 'deflections') else []

        if depths is None or len(depths) == 0:
            continue

        rgb = f'{int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)}'

        # BM plot
        fig.add_trace(go.Scatter(
            x=bm, y=depths, name=label,
            line=dict(color=color, width=2.5),
            fill='tozerox', fillcolor=f'rgba({rgb},0.08)',
            hovertemplate="Depth: %{y:.1f}m<br>BM: %{x:.1f} kN·m/m<extra></extra>",
        ), row=1, col=1)

        # SF plot
        fig.add_trace(go.Scatter(
            x=sf, y=depths, name=label,
            line=dict(color=color, width=2.5), showlegend=False,
            hovertemplate="Depth: %{y:.1f}m<br>SF: %{x:.1f} kN/m<extra></extra>",
        ), row=1, col=2)

        # Deflection plot
        fig.add_trace(go.Scatter(
            x=defl, y=depths, name=label,
            line=dict(color=color, width=2.5), showlegend=False,
            hovertemplate="Depth: %{y:.1f}m<br>δ: %{x:.1f} mm<extra></extra>",
        ), row=1, col=3)

        # ── Annotate peak values ──
        if bm is not None and len(bm) > 0:
            max_bm_val = max(abs(m) for m in bm)
            max_bm_idx = max(range(len(bm)), key=lambda i: abs(bm[i]))
            fig.add_annotation(
                x=bm[max_bm_idx], y=depths[max_bm_idx],
                text=f"<b>{bm[max_bm_idx]:.1f}</b><br>{depths[max_bm_idx]:.1f}m",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=color, font=dict(size=ANNOT_SIZE, color=color, family=FONT_FAMILY),
                bgcolor='rgba(255,255,255,0.9)', bordercolor=color, borderwidth=1, borderpad=3,
                ax=40, ay=-25, row=1, col=1,
            )

        if sf is not None and len(sf) > 0:
            max_sf_val = max(abs(s) for s in sf)
            max_sf_idx = max(range(len(sf)), key=lambda i: abs(sf[i]))
            fig.add_annotation(
                x=sf[max_sf_idx], y=depths[max_sf_idx],
                text=f"<b>{sf[max_sf_idx]:.1f}</b><br>{depths[max_sf_idx]:.1f}m",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=color, font=dict(size=ANNOT_SIZE, color=color, family=FONT_FAMILY),
                bgcolor='rgba(255,255,255,0.9)', bordercolor=color, borderwidth=1, borderpad=3,
                ax=40, ay=-25, row=1, col=2,
            )

        if defl is not None and len(defl) > 0:
            max_d_idx = max(range(len(defl)), key=lambda i: abs(defl[i]))
            fig.add_annotation(
                x=defl[max_d_idx], y=depths[max_d_idx],
                text=f"<b>{defl[max_d_idx]:.1f}</b><br>{depths[max_d_idx]:.1f}m",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=color, font=dict(size=ANNOT_SIZE, color=color, family=FONT_FAMILY),
                bgcolor='rgba(255,255,255,0.9)', bordercolor=color, borderwidth=1, borderpad=3,
                ax=40, ay=-25, row=1, col=3,
            )

    # ── Excavation level line ──
    for col in [1, 2, 3]:
        fig.add_hline(
            y=excavation_depth, line_dash="dash", line_color="#C0392B",
            line_width=1.5, row=1, col=col,
        )
    # Excavation label on first panel
    fig.add_annotation(
        x=0, y=excavation_depth,
        text=f"  Exc. {excavation_depth:.1f}m",
        showarrow=False, font=dict(size=13, color='#C0392B', family=FONT_FAMILY),
        xanchor='left', yanchor='bottom',
        row=1, col=1,
    )

    # ── Anchor level markers ──
    if anchors:
        for i, a in enumerate(anchors):
            for col in [1, 2, 3]:
                fig.add_hline(
                    y=a['level'], line_dash="dot", line_color=ANCHOR_COLOR,
                    line_width=1, row=1, col=col,
                )
            fig.add_annotation(
                x=0, y=a['level'],
                text=f"  A{i+1}: {a['level']:.1f}m",
                showarrow=False, font=dict(size=13, color=ANCHOR_COLOR),
                xanchor='left', yanchor='bottom', row=1, col=1,
            )

    # ── Styling ──
    fig.update_yaxes(autorange="reversed", title_text="<b>Depth (m)</b>", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=3)
    for col in [1, 2, 3]:
        fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, row=1, col=col)
        fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, row=1, col=col)

    _layout_defaults(fig, title=title, height=650)
    fig.update_layout(
        legend=dict(orientation="h", y=-0.10, x=0.5, xanchor='center'),
        margin=dict(t=90),
        title=dict(y=0.98),
    )

    # Style subplot titles and push them below main title
    for ann in fig.layout.annotations:
        if hasattr(ann, 'font'):
            ann.font.size = AXIS_SIZE
            ann.font.family = FONT_FAMILY
            ann.y = 0.92

    return fig


# ══════════════════════════════════════════════════════════════
# PRESSURE DISTRIBUTION PLOT
# ══════════════════════════════════════════════════════════════

def plot_pressure_distribution(
    depths: List[float],
    active: List[float],
    passive: List[float],
    excavation_depth: float,
    gwt_behind: float = None,
    gwt_front: float = None,
    pivot_depth: float = None,
    title: str = "Earth Pressure Distribution",
) -> go.Figure:
    """Professional pressure distribution with active/passive shading."""
    fig = go.Figure()

    # Active pressure (positive = toward excavation)
    fig.add_trace(go.Scatter(
        x=active, y=depths,
        name="Active (total)", fill='tozerox',
        fillcolor='rgba(139,69,19,0.12)',
        line=dict(color=ACTIVE_COLOR, width=2.5),
        hovertemplate="Depth: %{y:.1f}m<br>σₐ = %{x:.1f} kPa<extra></extra>",
    ))

    # Passive pressure (plotted negative for visual clarity)
    fig.add_trace(go.Scatter(
        x=[-p for p in passive], y=depths,
        name="Passive (total)", fill='tozerox',
        fillcolor='rgba(34,139,34,0.12)',
        line=dict(color=PASSIVE_COLOR, width=2.5),
        hovertemplate="Depth: %{y:.1f}m<br>σₚ = %{customdata:.1f} kPa<extra></extra>",
        customdata=passive,
    ))

    # Excavation level
    fig.add_hline(y=excavation_depth, line_dash="dash", line_color="#C0392B", line_width=1.5)
    fig.add_annotation(
        x=max(active) * 0.7, y=excavation_depth,
        text=f"<b>Excavation {excavation_depth:.1f}m</b>",
        showarrow=False, font=dict(size=ANNOT_SIZE, color='#C0392B'),
        yshift=-15, bgcolor='rgba(255,255,255,0.85)', bordercolor='#C0392B',
        borderwidth=1, borderpad=3,
    )

    # GWT
    if gwt_behind is not None:
        fig.add_hline(y=gwt_behind, line_dash="dot", line_color=GWT_COLOR, line_width=1.5)
        fig.add_annotation(
            x=max(active) * 0.9, y=gwt_behind,
            text=f"GWT {gwt_behind:.1f}m", showarrow=False,
            font=dict(size=13, color=GWT_COLOR), yshift=10,
        )

    # Pivot point
    if pivot_depth and pivot_depth > excavation_depth:
        fig.add_hline(y=pivot_depth, line_dash="dashdot", line_color='#8E44AD', line_width=1.2)
        fig.add_annotation(
            x=0, y=pivot_depth,
            text=f"Pivot {pivot_depth:.1f}m",
            showarrow=False, font=dict(size=13, color='#8E44AD'),
            yshift=12, xshift=50,
        )

    # Peak active annotation
    max_act = max(active)
    max_act_idx = active.index(max_act)
    fig.add_annotation(
        x=max_act, y=depths[max_act_idx],
        text=f"<b>{max_act:.1f} kPa</b>",
        showarrow=True, arrowhead=2, arrowcolor=ACTIVE_COLOR,
        font=dict(size=ANNOT_SIZE, color=ACTIVE_COLOR),
        ax=35, ay=-15,
        bgcolor='rgba(255,255,255,0.9)', bordercolor=ACTIVE_COLOR, borderwidth=1, borderpad=2,
    )

    fig.update_yaxes(autorange="reversed", title="<b>Depth (m)</b>")
    fig.update_xaxes(title="<b>Pressure (kPa)</b> — Active(+) / Passive(−)")
    _layout_defaults(fig, title=title, height=550)

    return fig


# ══════════════════════════════════════════════════════════════
# NET PRESSURE DIAGRAM
# ══════════════════════════════════════════════════════════════

def plot_net_pressure(
    depths: List[float],
    net_pressures: List[float],
    excavation_depth: float,
    pivot_depth: float = None,
    title: str = "Net Pressure Diagram",
) -> go.Figure:
    """Net pressure with positive/negative shading."""
    fig = go.Figure()

    # Split into positive and negative segments for dual-color fill
    pos_x = [max(n, 0) for n in net_pressures]
    neg_x = [min(n, 0) for n in net_pressures]

    fig.add_trace(go.Scatter(
        x=pos_x, y=depths, name="Net (+ve, drives wall)",
        fill='tozerox', fillcolor='rgba(139,69,19,0.15)',
        line=dict(color=ACTIVE_COLOR, width=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=neg_x, y=depths, name="Net (−ve, passive resists)",
        fill='tozerox', fillcolor='rgba(34,139,34,0.15)',
        line=dict(color=PASSIVE_COLOR, width=0.5),
    ))
    # Actual net line
    fig.add_trace(go.Scatter(
        x=net_pressures, y=depths, name="Net Pressure",
        line=dict(color=NET_COLOR, width=2.5),
        hovertemplate="Depth: %{y:.1f}m<br>Net: %{x:.1f} kPa<extra></extra>",
    ))

    fig.add_vline(x=0, line_color='#2C3E50', line_width=1)
    fig.add_hline(y=excavation_depth, line_dash="dash", line_color="#C0392B", line_width=1.5)
    fig.add_annotation(
        x=0, y=excavation_depth, text=f"  Exc. {excavation_depth:.1f}m",
        showarrow=False, font=dict(size=ANNOT_SIZE, color='#C0392B'),
        xanchor='left', yanchor='bottom',
    )

    if pivot_depth and pivot_depth > excavation_depth:
        fig.add_hline(y=pivot_depth, line_dash="dashdot", line_color='#8E44AD', line_width=1.2)
        fig.add_annotation(
            x=0, y=pivot_depth, text=f"  Pivot {pivot_depth:.1f}m",
            showarrow=False, font=dict(size=13, color='#8E44AD'),
            xanchor='left', yanchor='bottom',
        )

    fig.update_yaxes(autorange="reversed", title="<b>Depth (m)</b>")
    fig.update_xaxes(title="<b>Net Pressure (kPa)</b>")
    _layout_defaults(fig, title=title, height=480)

    return fig


# ══════════════════════════════════════════════════════════════
# UTILIZATION BAR CHART
# ══════════════════════════════════════════════════════════════

def plot_utilization(util_result, section_name: str, grade_name: str) -> go.Figure:
    """Professional utilization bar chart with threshold lines."""
    fig = go.Figure()

    checks = ['Bending', 'Shear', 'Combined', 'Interlock']
    vals = [
        util_result.utilization_bending,
        util_result.utilization_shear,
        util_result.utilization_combined,
        util_result.utilization_interlock,
    ]

    colors = []
    for v in vals:
        if v <= 0.70:
            colors.append('#27AE60')  # green
        elif v <= 0.90:
            colors.append('#2980B9')  # blue
        elif v <= 1.00:
            colors.append('#F39C12')  # amber
        else:
            colors.append('#E74C3C')  # red

    fig.add_trace(go.Bar(
        x=checks, y=vals, marker_color=colors,
        text=[f"{v:.0%}" for v in vals], textposition='outside',
        textfont=dict(size=AXIS_SIZE, family=FONT_FAMILY, color='#2C3E50'),
    ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="#E74C3C", line_width=2,
                  annotation_text="<b>Capacity (100%)</b>",
                  annotation_font=dict(size=13, color='#E74C3C'))
    fig.add_hline(y=0.9, line_dash="dot", line_color="#F39C12", line_width=1.5,
                  annotation_text="90%",
                  annotation_font=dict(size=13, color='#F39C12'))

    fig.update_yaxes(title="<b>Utilization Ratio</b>",
                     range=[0, max(1.3, max(vals) * 1.2)])
    _layout_defaults(fig, title=f"<b>{section_name}</b> — {grade_name}", height=380)
    fig.update_layout(margin=dict(t=70))

    return fig


# ══════════════════════════════════════════════════════════════
# STAGED EXCAVATION PLOTS
# ══════════════════════════════════════════════════════════════

STAGE_COLORS = [
    '#ABB2B9', '#85929E', '#5D6D7E',
    '#2E86C1', '#1A5276',
    '#28B463', '#1D8348',
    '#E67E22', '#CA6F1E',
    '#C0392B', '#922B21',
    '#8E44AD',
]


def plot_staged_envelope(
    staged_result,
    excavation_depth: float,
    anchors=None,
    title: str = "Design Envelope — All Stages",
) -> go.Figure:
    """BM / SF / Deflection envelope across all stages."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "<b>BM Envelope</b><br><sup>kN·m/m</sup>",
            "<b>SF Envelope</b><br><sup>kN/m</sup>",
            "<b>Deflection Envelope</b><br><sup>mm</sup>",
        ),
        shared_yaxes=True, horizontal_spacing=0.05,
    )

    depths = staged_result.envelope_depths

    fig.add_trace(go.Scatter(
        x=staged_result.envelope_bm_max, y=depths,
        name="BM Envelope", line=dict(color='#1A5276', width=3),
        fill='tozerox', fillcolor='rgba(26,82,118,0.15)',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=staged_result.envelope_sf_max, y=depths,
        name="SF Envelope", line=dict(color='#922B21', width=3),
        fill='tozerox', fillcolor='rgba(146,43,33,0.15)', showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=staged_result.envelope_defl_max, y=depths,
        name="Defl Envelope", line=dict(color='#6C3483', width=3),
        fill='tozerox', fillcolor='rgba(108,52,131,0.15)', showlegend=False,
    ), row=1, col=3)

    # Peak annotations
    for col, val, dep, stage, color, label in [
        (1, staged_result.design_bm, staged_result.design_bm_depth,
         staged_result.design_bm_stage, '#1A5276', 'kN·m/m'),
        (2, staged_result.design_sf, staged_result.design_sf_depth,
         staged_result.design_sf_stage, '#922B21', 'kN/m'),
        (3, staged_result.design_defl, staged_result.design_defl_depth,
         staged_result.design_defl_stage, '#6C3483', 'mm'),
    ]:
        fmt = f"{val:.0f}" if val > 10 else f"{val:.1f}"
        fig.add_annotation(
            x=val, y=dep,
            text=f"<b>{fmt}</b><br>{dep:.1f}m<br>Stage {stage}",
            showarrow=True, arrowhead=2, arrowcolor=color,
            font=dict(size=ANNOT_SIZE, color=color, family=FONT_FAMILY),
            bgcolor='rgba(255,255,255,0.92)', bordercolor=color,
            borderwidth=1, borderpad=3, ax=45, ay=-30, row=1, col=col,
        )

    for col in [1, 2, 3]:
        fig.add_hline(y=excavation_depth, line_dash="dash",
                     line_color="#C0392B", line_width=1.5, row=1, col=col)
    fig.add_annotation(
        x=0, y=excavation_depth, text=f"  Exc. {excavation_depth:.1f}m",
        showarrow=False, font=dict(size=13, color='#C0392B'),
        xanchor='left', yanchor='bottom', row=1, col=1,
    )
    if anchors:
        for i, a in enumerate(anchors):
            lev = a['level'] if isinstance(a, dict) else a.level
            for c in [1, 2, 3]:
                fig.add_hline(y=lev, line_dash="dot",
                             line_color=ANCHOR_COLOR, line_width=1, row=1, col=c)

    fig.update_yaxes(autorange="reversed", title_text="<b>Depth (m)</b>", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=3)
    _layout_defaults(fig, title=title, height=600)
    fig.update_layout(legend=dict(orientation="h", y=-0.10, x=0.5, xanchor='center'))
    return fig


def plot_staged_individual(
    staged_result,
    excavation_depth: float,
    anchors=None,
    title: str = "Per-Stage BM Diagrams",
) -> go.Figure:
    """Overlay BM curves from each stage."""
    fig = go.Figure()
    for sr in staged_result.stages:
        if sr.max_bm < 0.1:
            continue
        snum = sr.stage.stage_number
        cidx = min(snum, len(STAGE_COLORS) - 1)
        color = STAGE_COLORS[cidx]
        is_gov = (sr.max_bm == staged_result.design_bm)
        fig.add_trace(go.Scatter(
            x=sr.bending_moments, y=sr.depths,
            name=f"S{snum}: {sr.stage.description[:30]}",
            line=dict(color=color, width=3.0 if is_gov else 1.5),
            opacity=1.0 if is_gov else 0.5,
            hovertemplate=f"<b>Stage {snum}</b><br>"
                         f"Depth: %{{y:.1f}}m<br>BM: %{{x:.1f}}<extra></extra>",
        ))
    fig.add_hline(y=excavation_depth, line_dash="dash",
                 line_color="#C0392B", line_width=1.5)
    if anchors:
        for i, a in enumerate(anchors):
            lev = a['level'] if isinstance(a, dict) else a.level
            fig.add_hline(y=lev, line_dash="dot", line_color=ANCHOR_COLOR, line_width=1)
    fig.update_yaxes(autorange="reversed", title="<b>Depth (m)</b>")
    fig.update_xaxes(title="<b>Bending Moment (kN·m/m)</b>")
    _layout_defaults(fig, title=title, height=550)
    fig.update_layout(legend=dict(font=dict(size=12), y=0.5, x=1.02, xanchor='left'))
    return fig


def plot_stage_summary_bar(
    staged_result,
    title: str = "Peak Values per Stage",
) -> go.Figure:
    """Grouped bar chart: peak BM and SF at each stage."""
    stages = [r['stage'] for r in staged_result.summary]
    bms = [r['max_bm'] for r in staged_result.summary]
    sfs = [r['max_sf'] for r in staged_result.summary]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stages, y=bms, name="Max |BM| (kN·m/m)",
        marker_color='#1A5276', text=[f"{b:.0f}" for b in bms],
        textposition='outside', textfont=dict(size=13),
    ))
    fig.add_trace(go.Bar(
        x=stages, y=sfs, name="Max |SF| (kN/m)",
        marker_color='#922B21', text=[f"{s:.0f}" for s in sfs],
        textposition='outside', textfont=dict(size=13),
    ))
    gov_bm = staged_result.design_bm_stage
    fig.add_annotation(
        x=gov_bm, y=staged_result.design_bm,
        text="★ Governing", showarrow=True, arrowhead=2,
        font=dict(size=14, color='#C0392B', family=FONT_FAMILY),
        ax=0, ay=-30,
    )
    fig.update_xaxes(title="<b>Stage</b>", tickvals=stages, ticktext=[f"S{s}" for s in stages])
    fig.update_yaxes(title="<b>Force / Moment</b>")
    _layout_defaults(fig, title=title, height=400)
    fig.update_layout(barmode='group', legend=dict(orientation="h", y=-0.15, x=0.5, xanchor='center'))
    return fig


# ═══════════════════════════════════════════════════════════════
# COMBINED CROSS-SECTION (DeepEX-style)
# ═══════════════════════════════════════════════════════════════

def plot_combined_crosssection(
    layers, gwt_behind, gwt_front, excavation_depth, wall_toe,
    anchors, result, result_wind=None,
    surcharge=10.0, fos_passive=1.5,
    section_name="", Md=0, Vd=0,
    title="Combined Analysis — Cross-Section",
):
    """
    DeepEX-style 4-panel combined plot:
    Panel 1: Active/Passive pressure diagram
    Panel 2: Soil cross-section with wall, anchors, reactions
    Panel 3: Bending moment with capacity envelope
    Panel 4: Deflection with H/200 limit
    """
    import math

    max_depth = max(wall_toe + 1, max(L['bottom'] for L in layers))
    gamma_w = 9.81

    fig = make_subplots(
        rows=1, cols=4,
        column_widths=[0.28, 0.18, 0.28, 0.18],
        horizontal_spacing=0.03,
        subplot_titles=(
            "<b>Lateral Pressure (kPa)</b>",
            "<b>Cross-Section</b>",
            "<b>Bending Moment (kN·m/m)</b>",
            "<b>Deflection (mm)</b>",
        ),
    )

    # ─── PANEL 1: PRESSURE DIAGRAM ───
    dz = 0.1
    z_arr, pa_eff, pa_water, pa_total, pp_total = [], [], [], [], []
    z = 0.0
    while z <= wall_toe + 0.05:
        # Find layer at this depth
        phi, c, gamma = 30, 0, 18
        for L in layers:
            if L['top'] <= z < L['bottom']:
                phi = L.get('phi', 30); c = L.get('c', 0); gamma = L.get('gamma', 18)
                break

        Ka = math.tan(math.radians(45 - phi/2))**2
        Kp = math.tan(math.radians(45 + phi/2))**2

        # sigma_v effective
        sv = surcharge
        cum = 0
        for L in layers:
            lt, lb = L['top'], L['bottom']
            g = L.get('gamma', 18)
            if z <= lt: break
            z_in = min(z, lb) - lt
            if lb > gwt_behind:
                above = max(0, min(z_in, gwt_behind - lt))
                below = z_in - above
                sv += g * above + (g - gamma_w) * below
            else:
                sv += g * z_in
            cum = lb

        sig_ah = max(0, Ka * sv - 2 * c * math.sqrt(Ka))
        u_b = max(0, gamma_w * (z - gwt_behind))
        act = sig_ah + u_b

        # Passive below excavation
        pas = 0.0
        if z > excavation_depth:
            sv_f = 0
            for L in layers:
                lt, lb = L['top'], L['bottom']
                if lb <= excavation_depth: continue
                z_start = max(excavation_depth, lt)
                z_end = min(z, lb)
                if z_end <= z_start: continue
                g_f = L.get('gamma', 18)
                if z_end > gwt_front:
                    ab = max(0, min(gwt_front, z_end) - z_start)
                    bl = (z_end - z_start) - ab
                    sv_f += g_f * ab + (g_f - gamma_w) * bl
                else:
                    sv_f += g_f * (z_end - z_start)
            phi_f, c_f = phi, c
            Kp_f = math.tan(math.radians(45 + phi_f/2))**2
            u_f = max(0, gamma_w * (z - gwt_front))
            pas = Kp_f * sv_f + 2 * c_f * math.sqrt(Kp_f) + u_f

        z_arr.append(z)
        pa_eff.append(sig_ah)
        pa_water.append(u_b)
        pa_total.append(act)
        pp_total.append(pas / fos_passive if pas > 0 else 0)
        z += dz

    # Active fill
    fig.add_trace(go.Scatter(
        x=pa_eff, y=z_arr, mode='lines', name='Active (eff.)',
        line=dict(color='#E74C3C', width=1.5, dash='dash'),
        fill='tozerox', fillcolor='rgba(231,76,60,0.15)',
    ), row=1, col=1)

    # Water pressure
    fig.add_trace(go.Scatter(
        x=pa_total, y=z_arr, mode='lines', name='Active (total)',
        line=dict(color='#C0392B', width=2.5),
    ), row=1, col=1)

    # Passive
    fig.add_trace(go.Scatter(
        x=[-p for p in pp_total], y=z_arr, mode='lines', name='Passive/FOS',
        line=dict(color='#27AE60', width=2),
        fill='tozerox', fillcolor='rgba(39,174,96,0.12)',
    ), row=1, col=1)

    # Ka/Kp labels at layer midpoints
    for L in layers:
        if L['top'] >= wall_toe: continue
        phi = L.get('phi', 30)
        Ka = math.tan(math.radians(45 - phi/2))**2
        Kp = math.tan(math.radians(45 + phi/2))**2
        mid = (L['top'] + min(L['bottom'], wall_toe)) / 2
        max_p = max(pa_total) if pa_total else 50
        fig.add_annotation(
            x=max_p * 0.7, y=mid,
            text=f"Ka={Ka:.3f}<br>φ={phi:.0f}°",
            showarrow=False, font=dict(size=10, color='#7F8C8D'),
            bgcolor='rgba(255,255,255,0.85)', bordercolor='#BDC3C7',
            borderwidth=1, borderpad=3,
            row=1, col=1,
        )

    # Pressure values at layer boundaries
    for L in layers:
        bot = L['bottom']
        if bot > wall_toe: continue
        idx = min(int(bot / dz), len(pa_total) - 1)
        fig.add_annotation(
            x=pa_total[idx], y=bot,
            text=f"<b>{pa_total[idx]:.1f}</b>",
            showarrow=True, arrowhead=2, arrowcolor='#C0392B',
            font=dict(size=10, color='#C0392B'),
            ax=25, ay=0,
            row=1, col=1,
        )

    fig.update_xaxes(title_text="Pressure (kPa)", row=1, col=1)

    # ─── PANEL 2: CROSS-SECTION ───
    soil_colors = ['#D4A574', '#C4956A', '#A0886C', '#8B9467', '#7D8B6A', '#6B7D5E']

    for i, L in enumerate(layers):
        color = soil_colors[i % len(soil_colors)]
        if 'rock' in L['name'].lower(): color = '#8B8B6E'

        # Retained side (left: x=-1 to 0)
        fig.add_trace(go.Scatter(
            x=[-1, 0, 0, -1, -1],
            y=[L['top'], L['top'], L['bottom'], L['bottom'], L['top']],
            fill='toself', fillcolor=color, line=dict(width=0.5, color='#7F8C8D'),
            mode='lines', showlegend=False,
            hovertext=f"{L['name']}<br>γ={L.get('gamma',18)} kN/m³<br>φ={L.get('phi',30)}°",
            hoverinfo='text',
        ), row=1, col=2)

        # Excavation side (right: x=0 to 1)
        if L['bottom'] <= excavation_depth:
            # Fully excavated — white
            fig.add_trace(go.Scatter(
                x=[0, 1, 1, 0, 0],
                y=[L['top'], L['top'], L['bottom'], L['bottom'], L['top']],
                fill='toself', fillcolor='white', line=dict(width=0.5, color='#D5D8DC'),
                mode='lines', showlegend=False,
            ), row=1, col=2)
        elif L['top'] < excavation_depth:
            # Partially excavated
            fig.add_trace(go.Scatter(
                x=[0, 1, 1, 0, 0],
                y=[L['top'], L['top'], excavation_depth, excavation_depth, L['top']],
                fill='toself', fillcolor='white', line=dict(width=0.5, color='#D5D8DC'),
                mode='lines', showlegend=False,
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=[0, 1, 1, 0, 0],
                y=[excavation_depth, excavation_depth, L['bottom'], L['bottom'], excavation_depth],
                fill='toself', fillcolor=color, line=dict(width=0.5, color='#7F8C8D'),
                mode='lines', showlegend=False,
            ), row=1, col=2)
        else:
            fig.add_trace(go.Scatter(
                x=[0, 1, 1, 0, 0],
                y=[L['top'], L['top'], L['bottom'], L['bottom'], L['top']],
                fill='toself', fillcolor=color, line=dict(width=0.5, color='#7F8C8D'),
                mode='lines', showlegend=False,
            ), row=1, col=2)

        # Layer name
        mid = (L['top'] + min(L['bottom'], max_depth)) / 2
        fig.add_annotation(
            x=-0.5, y=mid, text=f"<b>{L['name']}</b>",
            showarrow=False, font=dict(size=9, color='white'),
            bgcolor='#2C3E50', borderpad=2, opacity=0.85,
            row=1, col=2,
        )

    # Wall
    fig.add_trace(go.Scatter(
        x=[-0.04, 0.04, 0.04, -0.04, -0.04],
        y=[0, 0, wall_toe, wall_toe, 0],
        fill='toself', fillcolor='#2C3E50', line=dict(width=1, color='#1C2833'),
        mode='lines', showlegend=False, hovertext=f"Wall L={wall_toe}m",
    ), row=1, col=2)

    # GWT
    fig.add_trace(go.Scatter(
        x=[-1, 1], y=[gwt_behind, gwt_behind],
        mode='lines', line=dict(color='#2E86C1', width=2, dash='dash'),
        name=f'GWT {gwt_behind}m', showlegend=False,
    ), row=1, col=2)
    fig.add_annotation(x=-0.9, y=gwt_behind, text=f"GWT {gwt_behind}m",
                       showarrow=False, font=dict(size=9, color='#2E86C1'),
                       yshift=-12, row=1, col=2)

    # Excavation level
    fig.add_trace(go.Scatter(
        x=[-1.1, 1.1], y=[excavation_depth, excavation_depth],
        mode='lines', line=dict(color='#C0392B', width=2, dash='dash'),
        showlegend=False,
    ), row=1, col=2)
    fig.add_annotation(x=0.5, y=excavation_depth,
                       text=f"<b>Exc. {excavation_depth}m</b>",
                       showarrow=False, font=dict(size=10, color='#C0392B'),
                       bgcolor='white', bordercolor='#C0392B', borderwidth=1,
                       yshift=-14, row=1, col=2)

    # Anchors with forces
    for i, a in enumerate(anchors):
        lvl = a['level'] if isinstance(a, dict) else a.level
        incl = a.get('incl', 20) if isinstance(a, dict) else getattr(a, 'inclination', 20)
        spc = a.get('spacing', 3.0) if isinstance(a, dict) else getattr(a, 'horizontal_spacing', 3.0)

        rxn = abs(result.anchor_reactions[i]) if i < len(result.anchor_reactions) else 0
        per_anc = rxn * spc

        length = 0.7
        dx = length * math.cos(math.radians(incl))
        dy = length * math.sin(math.radians(incl))

        fig.add_trace(go.Scatter(
            x=[0, -dx], y=[lvl, lvl + dy],
            mode='lines+markers',
            line=dict(color='#C0392B', width=3),
            marker=dict(symbol=['circle', 'arrow-left'], size=[5, 12], color='#C0392B'),
            showlegend=False,
        ), row=1, col=2)

        fig.add_annotation(
            x=-dx - 0.05, y=lvl + dy,
            text=f"<b>A{i+1}: {per_anc:.0f}kN</b><br>{rxn:.0f} kN/m",
            showarrow=False, font=dict(size=9, color='#C0392B'),
            bgcolor='#FADBD8', bordercolor='#C0392B', borderwidth=1, borderpad=3,
            xanchor='right', row=1, col=2,
        )

    # Toe reaction
    fig.add_annotation(
        x=0.4, y=wall_toe,
        text=f"R<sub>toe</sub>={abs(result.toe_reaction):.0f} kN/m",
        showarrow=True, arrowhead=2, arrowcolor='#8E44AD',
        font=dict(size=9, color='#8E44AD'),
        bgcolor='#E8DAEF', bordercolor='#8E44AD', borderwidth=1,
        ax=20, ay=20, row=1, col=2,
    )

    # Elevation labels
    for d in range(0, int(max_depth) + 1):
        fig.add_annotation(
            x=1.1, y=d, text=f"<sub>-{d}m</sub>",
            showarrow=False, font=dict(size=8, color='#ABB2B9'),
            xanchor='left', row=1, col=2,
        )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)

    # ─── PANEL 3: BENDING MOMENT ───
    depths = result.depths if hasattr(result, 'depths') else []
    bm = result.bending_moments if hasattr(result, 'bending_moments') else []

    if depths and bm:
        fig.add_trace(go.Scatter(
            x=bm, y=depths, mode='lines', name='BM',
            line=dict(color='#1A5276', width=2.5),
            fill='tozerox', fillcolor='rgba(26,82,118,0.1)',
        ), row=1, col=3)

        if result_wind:
            bm_w = result_wind.bending_moments
            fig.add_trace(go.Scatter(
                x=bm_w, y=result_wind.depths, mode='lines', name='BM (wind)',
                line=dict(color='#C0392B', width=2),
            ), row=1, col=3)

        # Capacity envelope
        if Md > 0:
            fig.add_trace(go.Scatter(
                x=[Md, Md], y=[0, max_depth], mode='lines', name=f'Md={Md:.0f}',
                line=dict(color='#C0392B', width=1.5, dash='dashdot'),
            ), row=1, col=3)
            fig.add_trace(go.Scatter(
                x=[-Md, -Md], y=[0, max_depth], mode='lines', name=f'-Md',
                line=dict(color='#C0392B', width=1.5, dash='dashdot'), showlegend=False,
            ), row=1, col=3)

        # Peak annotation
        import numpy as np
        bm_np = np.array(bm)
        max_idx = np.argmax(np.abs(bm_np))
        max_bm = bm_np[max_idx]
        max_d = depths[max_idx]
        util_pct = abs(max_bm) / Md * 100 if Md > 0 else 0

        fig.add_annotation(
            x=max_bm, y=max_d,
            text=f"<b>{max_bm:.1f}</b> kN·m/m<br>at {max_d:.1f}m",
            showarrow=True, arrowhead=2, arrowcolor='#1A5276',
            font=dict(size=11, color='#1A5276'),
            bgcolor='#D6EAF8', bordercolor='#1A5276', borderwidth=1,
            ax=-40, ay=-30, row=1, col=3,
        )

        # Utilization box
        if Md > 0:
            u_color = '#27AE60' if util_pct <= 100 else '#C0392B'
            u_text = f"{'PASS' if util_pct <= 100 else 'FAIL'} {util_pct:.0f}%"
            fig.add_annotation(
                x=0.95, y=0.02, xref='x3 domain', yref='y3 domain',
                text=f"<b>{u_text}</b>", showarrow=False,
                font=dict(size=12, color=u_color),
                bgcolor='white', bordercolor=u_color, borderwidth=2, borderpad=4,
            )

    # Excavation + anchor lines on BM panel
    for a in anchors:
        lvl = a['level'] if isinstance(a, dict) else a.level
        fig.add_hline(y=lvl, line=dict(color='#C0392B', width=0.8, dash='dot'),
                      row=1, col=3)

    fig.update_xaxes(title_text="BM (kN·m/m)", row=1, col=3)

    # ─── PANEL 4: DEFLECTION ───
    defl = result.deflections if hasattr(result, 'deflections') else []

    if depths and defl:
        fig.add_trace(go.Scatter(
            x=defl, y=depths, mode='lines', name='Deflection',
            line=dict(color='#6C3483', width=2.5),
            fill='tozerox', fillcolor='rgba(108,52,131,0.08)',
        ), row=1, col=4)

        # H/200 limit
        d_limit = excavation_depth * 1000 / 200
        fig.add_trace(go.Scatter(
            x=[d_limit, d_limit], y=[0, max_depth], mode='lines',
            name=f'H/200={d_limit:.0f}mm',
            line=dict(color='#E67E22', width=1.5, dash='dashdot'),
        ), row=1, col=4)

        # Peak deflection
        import numpy as np
        defl_np = np.array(defl)
        max_d_idx = np.argmax(np.abs(defl_np))
        fig.add_annotation(
            x=defl[max_d_idx], y=depths[max_d_idx],
            text=f"<b>{defl[max_d_idx]:.1f}</b> mm",
            showarrow=True, arrowhead=2, arrowcolor='#6C3483',
            font=dict(size=11, color='#6C3483'),
            bgcolor='#E8DAEF', bordercolor='#6C3483', borderwidth=1,
            ax=25, ay=-20, row=1, col=4,
        )

    fig.update_xaxes(title_text="Defl (mm)", row=1, col=4)

    # ─── COMMON: excavation line on all panels ───
    for col in [1, 3, 4]:
        fig.add_hline(y=excavation_depth, line=dict(color='#C0392B', width=1.5, dash='dash'),
                      row=1, col=col)

    # ─── GLOBAL LAYOUT ───
    for col in [1, 2, 3, 4]:
        fig.update_yaxes(autorange='reversed', range=[0, max_depth],
                         showgrid=True, gridcolor=GRID_COLOR, row=1, col=col)
    fig.update_yaxes(title_text="<b>Depth (m)</b>", row=1, col=1)

    _layout_defaults(fig, title=title, height=700)
    fig.update_layout(
        margin=dict(l=50, r=30, t=80, b=50),
        title=dict(y=0.98),
        showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor='center', font=dict(size=10)),
    )

    # Push subplot titles below main title
    for ann in fig.layout.annotations:
        if hasattr(ann, 'font') and ann.text and '<b>' in str(ann.text):
            if ann.y and ann.y > 0.9:
                ann.y = 0.93
                ann.font.size = 12

    return fig
