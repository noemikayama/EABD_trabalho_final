"""Dashboard descritivo sobre uso digital, comportamento e saúde mental.

A aplicação reúne três bases de dados e apresenta uma visão estática dos
principais indicadores da Geração Z. Os gráficos resumem distribuições,
correlações e diferenças entre métricas de comportamento adolescente e saúde
mental.

Execução:
    python app_dashboard_insights.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Caminhos construídos a partir da pasta do script para evitar dependência do
# diretório em que o comando de execução foi chamado.
BASE_DIR = Path(__file__).resolve().parent
GENZ_PATH = BASE_DIR / "genz_social_media_usage_1M (1).csv"
TEEN_PATH = BASE_DIR / "teen_behavior_patterns.csv"
MENTAL_PATH = BASE_DIR / "mental_health_trends.csv"

# Carregamento das três bases utilizadas pelo painel.
genz = pd.read_csv(GENZ_PATH)
teen = pd.read_csv(TEEN_PATH)
mental = pd.read_csv(MENTAL_PATH)

# Conversão da categoria ordinal para números, necessária para a correlação.
genz["addiction_level_num"] = genz["addiction_level"].map(
    {"Low": 1, "Medium": 2, "High": 3}
)

# Identidade visual compartilhada por cartões e gráficos.
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
    """Padroniza colunas numéricas para média zero e desvio padrão unitário.

    A transformação é feita em uma cópia, preservando os dados originais que
    alimentam os demais gráficos.
    """
    numeric = frame[columns].astype(float).copy()
    scaler = StandardScaler()
    standard_scaler = scaler.fit_transform(numeric)
    normalized = pd.DataFrame(
        standard_scaler,
        columns=numeric.columns,
        index=numeric.index,
    )
    return normalized


def normalize_min_max(frame, columns):
    """Retorna uma cópia das colunas numéricas normalizadas entre 0 e 1."""
    numeric = frame[columns].astype(float).copy()
    scaler = MinMaxScaler()
    min_max_scaler = scaler.fit_transform(numeric)
    normalized = pd.DataFrame(
        min_max_scaler,
        columns=numeric.columns,
        index=numeric.index,
    )
    return normalized


def correlation_strength(value):
    """Traduz o valor absoluto de uma correlação para uma faixa descritiva."""
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
    """Aplica o padrão visual do dashboard a uma figura Plotly."""
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
    """Cria um cartão de destaque com título, valor e texto explicativo."""
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


# Preparação dos indicadores
# Cada base é normalizada separadamente porque suas linhas representam pessoas
# diferentes. Os DataFrames originais permanecem intactos para os outros gráficos.
normalized_genz = normalize_for_correlation(genz, CORR_COLS)

teen_normalized_columns = [
    "academic_performance_score",
    "social_comparison_index",
    "body_image_anxiety_score",
    "peer_pressure_score",
]
normalized_teen = (
    normalize_min_max(teen, teen_normalized_columns)
)

mental_normalized_columns = [
    "anxiety_score",
    "depression_score",
    "stress_level",
    "loneliness_index",
    "self_esteem_score",
]
normalized_mental = (
    normalize_min_max(mental, mental_normalized_columns)
)

# Métricas resumidas exibidas no mapa de correlação e nos cartões.
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
combined_min_year = int(min(mental["year"].min(), teen["year"].min()))
combined_max_year = int(max(mental["year"].max(), teen["year"].max()))

# Histograma pré-agregado para evitar enviar um milhão de pontos ao navegador.
mental_counts, mental_bins = np.histogram(
    genz["mental_health_score"],
    bins=30,
)
fig_histogram = go.Figure(
    go.Bar(
        x=(mental_bins[:-1] + mental_bins[1:]) / 2,
        y=mental_counts,
        width=np.diff(mental_bins),
        marker_color=COLORS["primary"],
        hovertemplate=(
            "Saúde mental: %{x:.2f}<br>"
            "Participantes: %{y:,}<extra></extra>"
        ),
    )
)
fig_histogram.update_layout(
    title="Distribuição da saúde mental na Geração Z",
    xaxis_title="Pontuação de saúde mental",
    yaxis_title="Número de participantes",
    bargap=0.04,
)
fig_histogram = style_fig(fig_histogram)

# Matriz de correlação entre hábitos digitais e indicadores de bem-estar.
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

# Contagem por nível de dependência, mantendo uma ordem semântica fixa.
addiction_order = ["Low", "Medium", "High"]
addiction_labels = {
    "Low": "Baixo",
    "Medium": "Médio",
    "High": "Alto",
}
addiction_counts = (
    genz["addiction_level"]
    .value_counts()
    .reindex(addiction_order, fill_value=0)
)
fig_addiction = go.Figure(
    go.Bar(
        x=[addiction_labels[level] for level in addiction_order],
        y=addiction_counts.values,
        marker_color=[
            COLORS["success"],
            COLORS["accent"],
            COLORS["danger"],
        ],
        text=[f"{count:,}".replace(",", ".") for count in addiction_counts],
        textposition="outside",
        hovertemplate=(
            "Nível: %{x}<br>"
            "Participantes: %{y:,}<extra></extra>"
        ),
    )
)
fig_addiction.update_layout(
    title="Participantes por nível de dependência",
    xaxis_title="Nível de dependência",
    yaxis_title="Número de participantes",
    showlegend=False,
)
fig_addiction = style_fig(fig_addiction)

# Distribuição do tempo diário de uso, também resumida em 30 intervalos.
usage_counts, usage_bins = np.histogram(
    genz["daily_usage_hours"],
    bins=30,
)
fig_usage_histogram = go.Figure(
    go.Bar(
        x=(usage_bins[:-1] + usage_bins[1:]) / 2,
        y=usage_counts,
        width=np.diff(usage_bins),
        marker_color=COLORS["secondary"],
        hovertemplate=(
            "Uso diário: %{x:.2f} h<br>"
            "Participantes: %{y:,}<extra></extra>"
        ),
    )
)
fig_usage_histogram.update_layout(
    title="Distribuição das horas de uso diário",
    xaxis_title="Uso diário (horas)",
    yaxis_title="Número de participantes",
    bargap=0.04,
)
fig_usage_histogram = style_fig(fig_usage_histogram)

# Os nomes técnicos são traduzidos antes de reunir as bases em formato longo.
normalized_indicator_labels = {
    "anxiety_score": "Ansiedade",
    "depression_score": "Depressão",
    "stress_level": "Estresse",
    "loneliness_index": "Solidão",
    "self_esteem_score": "Autoestima",
    "academic_performance_score": "Desempenho acadêmico",
    "social_comparison_index": "Comparação social",
    "body_image_anxiety_score": "Ansiedade com imagem corporal",
    "peer_pressure_score": "Pressão dos pares",
}
# A escala comum de 0 a 1 permite comparar distribuições de indicadores que
# originalmente possuem unidades e amplitudes diferentes.
normalized_boxplot_data = pd.concat(
    [
        normalized_mental.rename(columns=normalized_indicator_labels)
        .melt(
            var_name="indicator",
            value_name="normalized_value",
        )
        .assign(source="Tendências de saúde mental"),
        normalized_teen.rename(columns=normalized_indicator_labels)
        .melt(
            var_name="indicator",
            value_name="normalized_value",
        )
        .assign(source="Comportamento adolescente"),
    ],
    ignore_index=True,
)
fig_normalized_boxplots = px.box(
    normalized_boxplot_data,
    x="indicator",
    y="normalized_value",
    color="source",
    points=False,
    title=(
        "Distribuição normalizada dos indicadores de saúde e comportamento "
        f"de {combined_min_year} a {combined_max_year}"
    ),
    labels={
        "indicator": "Indicador",
        "normalized_value": "Valor normalizado",
        "source": "Base de dados",
    },
    color_discrete_map={
        "Tendências de saúde mental": COLORS["primary"],
        "Comportamento adolescente": COLORS["secondary"],
    },
)
fig_normalized_boxplots.update_traces(
    hovertemplate=(
        "Indicador: %{x}<br>"
        "Valor normalizado: %{y:.3f}<extra></extra>"
    ),
)
fig_normalized_boxplots.update_layout(
    yaxis_range=[0, 1],
)
fig_normalized_boxplots.update_xaxes(tickangle=-25)
fig_normalized_boxplots = style_fig(fig_normalized_boxplots)

# Configuração e composição visual da aplicação.
app = Dash(__name__)
app.title = "Dashboard de Insights"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(
                    "DASHBOARD DE INSIGHTS",
                    style={
                        "margin": "0 0 6px",
                        "fontWeight": "bold",
                        "letterSpacing": "1.5px",
                        "color": COLORS["primary"],
                    },
                ),
                html.H1(
                    "Geração Z em Rede: Uso Digital, Comportamento e Saúde Mental",
                    style={"margin": "0", "fontSize": "34px"},
                ),
                html.P(
                    (
                        "Uma leitura integrada de como os hábitos digitais se relacionam "
                        "com o bem-estar, a saúde mental e o comportamento adolescente, "
                        "revelando padrões, diferenças e conexões entre os dados."
                    ),
                    style={
                        "margin": "9px 0 0",
                        "color": COLORS["muted"],
                        "fontSize": "16px",
                    },
                ),
            ],
            style={"marginBottom": "22px"},
        ),
        html.Div(
            children=[
                html.Div(dcc.Graph(figure=fig_histogram), className="chart-card"),
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
        html.Div(
            children=[
                html.Div(
                    dcc.Graph(figure=fig_addiction),
                    className="chart-card",
                ),
                html.Div(
                    dcc.Graph(figure=fig_usage_histogram),
                    className="chart-card",
                ),
            ],
            className="graphs-grid dependency-grid",
        ),
        html.Div(
            dcc.Graph(figure=fig_normalized_boxplots),
            className="chart-card normalized-boxplots-chart",
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

# CSS responsivo incorporado ao documento HTML servido pelo Dash.
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
                margin-bottom: 18px;
            }}
            .dependency-grid {{
                margin-top: 0;
            }}
            .normalized-boxplots-chart {{
                margin-top: 18px;
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
    # Porta distinta da aplicação interativa para permitir executar ambas juntas.
    app.run(debug=True, port=8051)
