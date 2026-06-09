from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
GENZ_PATH = BASE_DIR / "genz_social_media_usage_1M (1).csv"
TEEN_PATH = BASE_DIR / "teen_behavior_patterns.csv"
MENTAL_PATH = BASE_DIR / "mental_health_trends.csv"

genz = pd.read_csv(GENZ_PATH)
teen = pd.read_csv(TEEN_PATH) if TEEN_PATH.exists() else pd.DataFrame()
mental = pd.read_csv(MENTAL_PATH) if MENTAL_PATH.exists() else pd.DataFrame()
genz["addiction_level_num"] = genz["addiction_level"].map(
    {"Low": 1, "Medium": 2, "High": 3}
)
genz_sample = genz.sample(n=min(35000, len(genz)), random_state=42)

COLORS = {
    "background": "#F6F8FB",
    "card": "#FFFFFF",
    "text": "#1F2937",
    "muted": "#6B7280",
    "primary": "#4F46E5",
    "secondary": "#06B6D4",
    "accent": "#F97316",
    "danger": "#E11D48",
    "success": "#10B981",
    "border": "#E5E7EB",
}

PLOTLY_TEMPLATE = "plotly_white"
DISCRETE_SEQUENCE = [
    "#4F46E5",
    "#06B6D4",
    "#F97316",
    "#10B981",
    "#E11D48",
    "#8B5CF6",
]

CORR_COLS = [
    "daily_usage_hours",
    "addiction_level_num",
    "mental_health_score",
    "num_platforms_used",
    "avg_session_minutes",
    "night_usage",
    "screen_time_before_sleep",
]
CORR_LABELS = {
    "daily_usage_hours": "Uso diário",
    "addiction_level_num": "Nível de dependência",
    "mental_health_score": "Saúde mental",
    "num_platforms_used": "Nº de plataformas",
    "avg_session_minutes": "Duração da sessão",
    "night_usage": "Uso noturno",
    "screen_time_before_sleep": "Tela antes de dormir",
}


def normalize_for_correlation(frame, columns):
    """Cria uma cópia padronizada sem alterar os dados usados nos gráficos."""
    numeric = frame[columns].astype(float).copy()
    scaler = StandardScaler()
    standard_scaler = scaler.fit_transform(numeric)
    normalized = pd.DataFrame(
        standard_scaler,
        columns=numeric.columns,
        index=numeric.index,
    )
    return normalized


def correlation_strength(value):
    absolute = abs(value)
    if np.isclose(absolute, 1.0):
        return "perfeita"
    if absolute >= 0.90:
        return "muito forte"
    if absolute >= 0.70:
        return "forte"
    if absolute >= 0.40:
        return "moderada"
    if absolute >= 0.20:
        return "fraca"
    if absolute >= 0.01:
        return "muito fraca"
    return "ausente"


def style_fig(fig):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Arial"),
        title=dict(font=dict(size=20), x=0.02),
        margin=dict(l=45, r=30, t=75, b=65),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF2F7")
    return fig


def insight_card(title, value, description, color):
    return html.Div(
        children=[
            html.P(title, style={"margin": "0", "fontSize": "13px", "fontWeight": "bold", "color": COLORS["muted"]}),
            html.H3(value, style={"margin": "9px 0 6px", "fontSize": "24px", "color": color}),
            html.P(description, style={"margin": "0", "fontSize": "13px", "lineHeight": "1.45"}),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderTop": f"4px solid {color}",
            "borderRadius": "16px",
            "padding": "16px",
            "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.05)",
            "minHeight": "140px",
        },
    )


# Cada base é normalizada separadamente porque suas linhas representam pessoas
# diferentes. Os DataFrames originais permanecem intactos para os outros gráficos.
normalized_genz = normalize_for_correlation(genz, CORR_COLS)

teen_corr_columns = [
    "year",
    "academic_performance_score",
    "social_comparison_index",
    "body_image_anxiety_score",
    "peer_pressure_score",
    "suicide_risk_indicator",
]
normalized_teen = (
    normalize_for_correlation(teen, teen_corr_columns)
    if not teen.empty
    else pd.DataFrame()
)

mental_corr_source = mental.copy()
if not mental_corr_source.empty:
    mental_corr_source["mental_health_risk_num"] = mental_corr_source[
        "mental_health_risk"
    ].map({"Low": 1, "Medium": 2, "High": 3})
mental_corr_columns = [
    "year",
    "anxiety_score",
    "depression_score",
    "stress_level",
    "loneliness_index",
    "therapy_access",
    "medication_usage",
    "self_esteem_score",
    "mental_health_risk_num",
]
normalized_mental = (
    normalize_for_correlation(mental_corr_source, mental_corr_columns)
    if not mental_corr_source.empty
    else pd.DataFrame()
)

corr = normalized_genz.corr(numeric_only=True)
usage_addiction_corr = corr.loc[
    "daily_usage_hours",
    "addiction_level_num",
]
usage_mental_corr = corr.loc["daily_usage_hours", "mental_health_score"]
min_age = int(genz["age"].min())
max_age = int(genz["age"].max())
teen_mental = genz.loc[genz["age"].between(13, 19), "mental_health_score"].mean()
total_records = len(genz) + len(teen) + len(mental)

fig_scatter = px.scatter(
    genz_sample,
    x="daily_usage_hours",
    y="mental_health_score",
    color="primary_platform",
    size="avg_session_minutes",
    opacity=0.42,
    color_discrete_sequence=DISCRETE_SEQUENCE,
    title="Uso diário de redes sociais e saúde mental",
    labels={
        "daily_usage_hours": "Uso diário (horas)",
        "mental_health_score": "Pontuação de saúde mental",
        "primary_platform": "Plataforma",
        "avg_session_minutes": "Duração média da sessão",
    },
)
fig_scatter = style_fig(fig_scatter)

fig_heatmap = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale=["#06B6D4", "#FFFFFF", "#4F46E5"],
    zmin=-1,
    zmax=1,
    title="Mapa completo de correlações de uso e bem-estar",
    labels={"color": "Correlação"},
)
fig_heatmap.update_xaxes(
    tickmode="array",
    tickvals=CORR_COLS,
    ticktext=[CORR_LABELS[column] for column in CORR_COLS],
    tickangle=-25,
)
fig_heatmap.update_yaxes(
    tickmode="array",
    tickvals=CORR_COLS,
    ticktext=[CORR_LABELS[column] for column in CORR_COLS],
)
fig_heatmap = style_fig(fig_heatmap)

app = Dash(__name__)
app.title = "Insights sobre Redes Sociais e Bem-estar"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P("DASHBOARD DE INSIGHTS", style={"margin": "0 0 6px", "fontWeight": "bold", "letterSpacing": "1.5px", "color": COLORS["primary"]}),
                html.H1("Insights sobre Redes Sociais e Bem-estar", style={"margin": "0", "fontSize": "34px"}),
                html.P(
                    "A dispersão mostra os registros individuais e o mapa resume as relações lineares entre as variáveis.",
                    style={"margin": "9px 0 0", "color": COLORS["muted"], "fontSize": "16px"},
                ),
            ],
            style={"marginBottom": "22px"},
        ),
        html.Div(
            children=[
                html.Div(dcc.Graph(figure=fig_scatter), className="chart-card"),
                html.Div(dcc.Graph(figure=fig_heatmap), className="chart-card"),
            ],
            className="graphs-grid",
        ),
        html.Div(
            children=[
                insight_card(
                    "Intervalo de idade",
                    f"{min_age} a {max_age} anos",
                    "Menor e maior idade encontradas na base Gen Z.",
                    COLORS["accent"],
                ),
                insight_card(
                    "Saúde mental dos adolescentes",
                    f"{teen_mental:.2f}/10",
                    "Média dos participantes de 13 a 19 anos na base Gen Z.",
                    COLORS["success"],
                ),
                insight_card(
                    "Total de registros avaliados",
                    f"{total_records:,}".replace(",", "."),
                    (
                        f"{len(genz):,} da base de uso de rede social da Gen Z, {len(teen):,} da base de saúde mental de adolescentes. "
                        f"{len(mental):,} da base de tendências de saúde mental."
                    ).replace(",", "."),
                    COLORS["secondary"],
                ),
                insight_card(
                    "Correlações com o uso diário",
                    (
                        f"r = {usage_addiction_corr:.2f} | "
                        f"r = {usage_mental_corr:.2f}"
                    ),
                    (
                        f"Uso diário e nível de dependência apresentam correlação "
                        f"{correlation_strength(usage_addiction_corr)} e positiva "
                        f"(r = {usage_addiction_corr:.2f}). Uso diário e saúde mental "
                        f"apresentam correlação {correlation_strength(usage_mental_corr)} "
                        f"e negativa (r = {usage_mental_corr:.2f}). "
                        "Correlação não comprova causalidade."
                    ),
                    COLORS["primary"],
                ),
            ],
            className="insight-grid",
        ),
    ],
    style={
        "minHeight": "100vh",
        "padding": "24px",
        "backgroundColor": COLORS["background"],
        "fontFamily": "Arial, sans-serif",
        "color": COLORS["text"],
    },
)

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            .graphs-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 16px;
                margin-bottom: 18px;
            }}
            .chart-card {{
                min-width: 0;
                overflow: hidden;
                padding: 8px;
                background-color: {COLORS["card"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 20px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            }}
            .insight-grid {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 14px;
            }}
            @media (max-width: 1050px) {{
                .graphs-grid {{ grid-template-columns: 1fr; }}
                .insight-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            }}
            @media (max-width: 650px) {{
                .insight-grid {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=8051)
