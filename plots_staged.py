"""
Staged excavation plots - envelope diagrams and per-stage summary.

"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ANCHOR_COLOR = '#B22222'
FONT_FAMILY = "Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
ANNOT_SIZE = 14
GRID_COLOR = '#E5E5E5'

STAGE_PALETTE = [
    '#BDC3C7', '#95A5A6', '#7FB3D8', '#5DADE2', '#2E86C1',
    '#1A5276', '#D4AC0D', '#D68910', '#CA6F1E', '#A04000',
    '#922B21', '#641E16',
]


def _layout_defaults(fig, title="", height=550):
    fig.update_layout(
        height=height, template="plotly_white",
        title=dict(text=title, font=dict(size=20, family=FONT_FAMILY, color='#1C2833')),
        font=dict(family=FONT_FAMILY, size=15, color='#2C3E50'),
        plot_bgcolor='#FAFBFC', paper_bgcolor='white',
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(font=dict(size=ANNOT_SIZE), bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#D5D8DC', borderwidth=1),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, gridwidth=0.5, tickfont=dict(size=15), title_font=dict(size=16))
    fig.update_yaxes(gridcolor=GRID_COLOR, gridwidth=0.5, tickfont=dict(size=15), title_font=dict(size=16))
    return fig


def plot_staged_envelope(staged_result, excavation_depth, anchors=None,
                         title="Staged Analysis - BM / SF / Deflection Envelope"):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "<b>Bending Moment</b><br><sup>kN.m/m</sup>",
            "<b>Shear Force</b><br><sup>kN/m</sup>",
            "<b>Deflection</b><br><sup>mm</sup>",
        ),
        shared_yaxes=True, horizontal_spacing=0.05,
    )

    for idx, sr in enumerate(staged_result.stages):
        if sr.max_bm < 0.01 and sr.max_sf < 0.01:
            continue
        color = STAGE_PALETTE[idx % len(STAGE_PALETTE)]
        label = "S%d: %.0fm" % (sr.stage.stage_number, sr.stage.excavation_depth)

        fig.add_trace(go.Scatter(
            x=sr.bending_moments, y=sr.depths, name=label,
            line=dict(color=color, width=1.2), opacity=0.6, showlegend=True,
            hovertemplate="Stage %d<br>Depth: %%{y:.1f}m<br>BM: %%{x:.1f}<extra></extra>" % sr.stage.stage_number,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sr.shear_forces, y=sr.depths, name=label,
            line=dict(color=color, width=1.2), opacity=0.6, showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=sr.deflections, y=sr.depths, name=label,
            line=dict(color=color, width=1.2), opacity=0.6, showlegend=False,
        ), row=1, col=3)

    env_d = staged_result.envelope_depths
    fig.add_trace(go.Scatter(
        x=staged_result.envelope_bm_max, y=env_d,
        name="<b>ENVELOPE</b>", line=dict(color='#1C2833', width=3.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=staged_result.envelope_sf_max, y=env_d,
        name="Envelope", line=dict(color='#1C2833', width=3.5), showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=staged_result.envelope_defl_max, y=env_d,
        name="Envelope", line=dict(color='#1C2833', width=3.5), showlegend=False,
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=staged_result.envelope_bm_min, y=env_d,
        name="Env (min)", line=dict(color='#1C2833', width=2, dash='dash'),
        opacity=0.5, showlegend=False,
    ), row=1, col=1)

    peak_data = [
        (staged_result.design_bm, staged_result.design_bm_depth, staged_result.design_bm_stage, 1),
        (staged_result.design_sf, staged_result.design_sf_depth, staged_result.design_sf_stage, 2),
        (staged_result.design_defl, staged_result.design_defl_depth, staged_result.design_defl_stage, 3),
    ]
    for val, dep, stg, col_idx in peak_data:
        fig.add_annotation(
            x=val, y=dep,
            text="<b>%.1f</b><br>%.1fm<br>Stage %d" % (val, dep, stg),
            showarrow=True, arrowhead=2, arrowcolor='#C0392B',
            font=dict(size=ANNOT_SIZE, color='#C0392B', family=FONT_FAMILY),
            bgcolor='rgba(255,255,255,0.9)', bordercolor='#C0392B',
            borderwidth=1, borderpad=3, ax=45, ay=-30,
            row=1, col=col_idx,
        )

    for col in [1, 2, 3]:
        fig.add_hline(y=excavation_depth, line_dash="dash", line_color="#C0392B", line_width=1.5, row=1, col=col)
    fig.add_annotation(x=0, y=excavation_depth,
                       text="  Final exc. %.1fm" % excavation_depth,
                       showarrow=False, font=dict(size=13, color='#C0392B'),
                       xanchor='left', yanchor='bottom', row=1, col=1)

    if anchors:
        for i, a in enumerate(anchors):
            lvl = a['level'] if isinstance(a, dict) else a.level
            for col in [1, 2, 3]:
                fig.add_hline(y=lvl, line_dash="dot", line_color=ANCHOR_COLOR, line_width=1, row=1, col=col)

    fig.update_yaxes(autorange="reversed", title_text="<b>Depth (m)</b>", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=3)
    _layout_defaults(fig, title=title, height=700)
    fig.update_layout(
        legend=dict(orientation="h", y=-0.10, x=0.5, xanchor='center', font=dict(size=12)),
        margin=dict(t=90),
        title=dict(y=0.98),
    )
    # Push subplot titles below main title
    for ann in fig.layout.annotations:
        if hasattr(ann, 'font') and ann.text and '<b>' in str(ann.text):
            ann.y = 0.92
            ann.font.size = 16
    return fig


def plot_stage_summary_bars(staged_result, title="Per-Stage Peak Values"):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("<b>Max BM (kN.m/m)</b>", "<b>Max SF (kN/m)</b>", "<b>Max Defl (mm)</b>"),
        horizontal_spacing=0.08,
    )

    stages = [s for s in staged_result.summary if s['exc_depth'] > 0]
    labels = ["S%d" % s['stage'] for s in stages]
    bm_vals = [s['max_bm'] for s in stages]
    sf_vals = [s['max_sf'] for s in stages]
    defl_vals = [s['max_defl'] for s in stages]

    bm_colors = ['#C0392B' if abs(v - staged_result.design_bm) < 0.5 else '#2E86C1' for v in bm_vals]
    sf_colors = ['#C0392B' if abs(v - staged_result.design_sf) < 0.5 else '#2E86C1' for v in sf_vals]
    defl_colors = ['#C0392B' if abs(v - staged_result.design_defl) < 0.5 else '#2E86C1' for v in defl_vals]

    fig.add_trace(go.Bar(x=labels, y=bm_vals, marker_color=bm_colors,
        text=["%.0f" % v for v in bm_vals], textposition='outside', textfont=dict(size=13)), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=sf_vals, marker_color=sf_colors,
        text=["%.0f" % v for v in sf_vals], textposition='outside', textfont=dict(size=13)), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=defl_vals, marker_color=defl_colors,
        text=["%.1f" % v for v in defl_vals], textposition='outside', textfont=dict(size=13)), row=1, col=3)

    _layout_defaults(fig, title=title, height=380)
    fig.update_layout(showlegend=False)
    return fig
