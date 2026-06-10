"""Dashboard interativo sobre uso digital, comportamento e saúde mental.

Os filtros de país e gênero atualizam indicadores e gráficos construídos com
três bases de dados. A aplicação explora relações entre tempo de uso, nível de
dependência, saúde mental, desempenho acadêmico e comportamento adolescente.

Execução:
    python app_dashboard_interactive.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.preprocessing import StandardScaler

# Caminhos relativos ao próprio script tornam a leitura dos CSVs independente
# do diretório usado para iniciar a aplicação.
BASE_DIR = Path(__file__).resolve().parent
GENZ_PATH = BASE_DIR / "genz_social_media_usage_1M (1).csv"
TEEN_PATH = BASE_DIR / "teen_behavior_patterns.csv"
MENTAL_PATH = BASE_DIR / "mental_health_trends.csv"

# Carregamento das bases analisadas pelo dashboard.
genz = pd.read_csv(GENZ_PATH)
teen = pd.read_csv(TEEN_PATH)
mental = pd.read_csv(MENTAL_PATH)

# Colunas auxiliares usadas em legendas, correlações e eixos categóricos.
genz["night_usage_label"] = np.where(
    genz["night_usage"] == 1,
    "Usa à noite",
    "Não usa à noite",
)
# A amostra reduz o custo de renderização do gráfico de dispersão sem alterar
# os cálculos dos KPIs e das correlações, que continuam usando a base completa.
genz_sample = genz.sample(n=min(35000, len(genz)), random_state=42)

genz["addiction_level_num"] = genz["addiction_level"].map(
    {"Low": 1, "Medium": 2, "High": 3}
)
mental["mental_health_risk_num"] = mental["mental_health_risk"].map(
    {"Low": 1, "Medium": 2, "High": 3}
)

# Identidade visual compartilhada pelos componentes da aplicação.
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

TRANSLATIONS = {
    "Male": "Masculino",
    "Female": "Feminino",
    "Other": "Outro",
    "Brazil": "Brasil",
    "USA": "Estados Unidos",
    "UK": "Reino Unido",
    "India": "Índia",
    "Germany": "Alemanha",
    "Australia": "Austrália",
    "Canada": "Canadá",
}


def options_from(series):
    """Converte valores de uma coluna em opções traduzidas para um Dropdown."""
    values = sorted(series.unique())
    return [
        {"label": TRANSLATIONS.get(value, str(value)), "value": value}
        for value in values
    ]


def filter_genz(countries, genders, platforms, source=None):
    """Filtra a base Gen Z, ou uma amostra dela, por país e gênero."""
    filtered = genz if source is None else source
    if countries:
        filtered = filtered[filtered["country"].isin(countries)]
    if genders:
        filtered = filtered[filtered["gender"].isin(genders)]
    return filtered


def filter_teen(countries, genders, platforms):
    """Filtra os dados de comportamento adolescente por país e gênero."""
    if teen.empty:
        return teen
    filtered = teen
    if countries:
        filtered = filtered[filtered["country"].isin(countries)]
    if genders:
        filtered = filtered[filtered["gender"].isin(genders)]
    return filtered


def filter_mental(countries, genders):
    """Filtra a base de tendências de saúde mental por país e gênero."""
    if mental.empty:
        return mental
    filtered = mental
    if countries:
        filtered = filtered[filtered["country"].isin(countries)]
    if genders:
        filtered = filtered[filtered["gender"].isin(genders)]
    return filtered


def normalize_for_correlation(frame, columns):
    """Padroniza colunas numéricas sem modificar o DataFrame de origem."""
    numeric = frame[columns].astype(float).copy()
    scaler = StandardScaler()
    standard_scaler = scaler.fit_transform(numeric)
    normalized = pd.DataFrame(
        standard_scaler,
        columns=numeric.columns,
        index=numeric.index,
    )
    return normalized


def style_fig(fig):
    """Aplica cores, fontes, margens e grades padronizadas a uma figura."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Arial"),
        title=dict(font=dict(size=18), x=0.02),
        margin=dict(l=40, r=30, t=70, b=55),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF2F7")
    return fig


def empty_figure(title, message="Não há dados disponíveis para os filtros selecionados."):
    """Cria um gráfico vazio com uma mensagem contextual para o usuário."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=15, color=COLORS["muted"]),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def kpi_card(title, value, subtitle, color):
    """Monta um cartão de KPI com valor, descrição e cor de destaque."""
    return html.Div(
        children=[
            html.P(title, style={"margin": "0", "fontSize": "13px", "color": COLORS["muted"]}),
            html.H2(value, style={"margin": "8px 0 4px", "fontSize": "28px", "color": color}),
            html.P(subtitle, style={"margin": "0", "fontSize": "12px", "color": COLORS["muted"]}),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "16px",
            "padding": "18px",
            "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.06)",
            "minHeight": "110px",
        },
    )


# Configuração e composição visual da aplicação.
app = Dash(__name__)
app.title = "Dashboard Interativo"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(
                    "DASHBOARD INTERATIVO",
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
                html.Div(
                    children=[
                        html.Label("Países", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="country-filter",
                            options=options_from(genz["country"]),
                            value=["Brazil", "USA", "UK"],
                            multi=True,
                            placeholder="Todos os países",
                        ),
                    ],
                    style={"flex": "2", "minWidth": "240px"},
                ),
                html.Div(
                    children=[
                        html.Label("Gêneros", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="gender-filter",
                            options=options_from(genz["gender"]),
                            multi=True,
                            placeholder="Todos os gêneros",
                        ),
                    ],
                    style={"flex": "1.3", "minWidth": "220px"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "16px",
                "flexWrap": "wrap",
                "backgroundColor": COLORS["card"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "18px",
                "padding": "18px",
                "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.05)",
                "marginBottom": "18px",
            },
        ),
        html.Div(
            id="kpi-row",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(210px, 1fr))",
                "gap": "16px",
                "marginBottom": "18px",
            },
        ),
        html.Div(
            id="insight-box",
            style={
                "background": f"linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%)",
                "color": "white",
                "borderRadius": "18px",
                "padding": "18px 22px",
                "boxShadow": "0 8px 22px rgba(79, 70, 229, 0.20)",
                "marginBottom": "18px",
                "fontSize": "16px",
            },
        ),
        html.Div(
            children=[
                html.Div(dcc.Graph(id="scatter-usage-mental"), className="chart-card"),
                html.Div(dcc.Graph(id="usage-addiction"), className="chart-card"),
                html.Div(dcc.Graph(id="night-academic"), className="chart-card"),
                html.Div(dcc.Graph(id="strong-correlations"), className="chart-card"),
                html.Div(dcc.Graph(id="suicide-risk-platform"), className="chart-card"),
                html.Div(dcc.Graph(id="usage-addiction-mental"), className="chart-card"),
            ],
            className="charts-grid",
        ),
    ],
    style={
        "backgroundColor": COLORS["background"],
        "minHeight": "100vh",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
        "color": COLORS["text"],
    },
)

# CSS responsivo incorporado ao HTML principal servido pelo Dash.
app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            .charts-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 16px;
            }}
            .chart-card {{
                min-width: 0;
                padding: 8px;
                background-color: {COLORS["card"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 18px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            }}
            .chart-wide {{
                grid-column: span 2;
            }}
            @media (max-width: 1050px) {{
                .charts-grid {{
                    grid-template-columns: 1fr;
                }}
                .chart-wide {{
                    grid-column: span 1;
                }}
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


@app.callback(
    Output("kpi-row", "children"),
    Output("insight-box", "children"),
    Output("scatter-usage-mental", "figure"),
    Output("usage-addiction", "figure"),
    Output("night-academic", "figure"),
    Output("strong-correlations", "figure"),
    Output("suicide-risk-platform", "figure"),
    Output("usage-addiction-mental", "figure"),
    Input("country-filter", "value"),
    Input("gender-filter", "value"),
)
def update_dashboard(countries, genders):
    """Atualiza KPIs, insight textual e gráficos conforme os filtros.

    Os filtros são aplicados separadamente às três bases. Quando um gráfico
    combina fontes diferentes, os registros são agregados por dimensões em
    comum, evitando associar diretamente indivíduos sem uma chave compartilhada.
    """
    filtered = filter_genz(countries, genders, None)
    filtered_teen = filter_teen(countries, genders, None)
    filtered_mental = filter_mental(countries, genders)

    if filtered.empty:
        empty = empty_figure("Sem dados")
        return (
            [kpi_card("Sem dados", "-", "Altere os filtros", COLORS["danger"])],
            "Não há registros para os filtros selecionados.",
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        )

    # Indicadores gerais calculados sobre todos os registros selecionados.
    avg_usage = filtered["daily_usage_hours"].mean()
    avg_mental = filtered["mental_health_score"].mean()
    night_share = filtered["night_usage"].mean() * 100
    avg_sleep = filtered["screen_time_before_sleep"].mean()
    kpis = [
        kpi_card("Uso diário médio", f"{avg_usage:.2f} h", "Uso de redes sociais", COLORS["primary"]),
        kpi_card("Saúde mental média", f"{avg_mental:.2f}/10", "Valores maiores indicam melhor pontuação", COLORS["success"]),
        kpi_card("Uso durante a noite", f"{night_share:.1f}%", "Usuários ativos à noite", COLORS["accent"]),
        kpi_card("Tela antes de dormir", f"{avg_sleep:.1f} min", "Tempo médio antes de dormir", COLORS["secondary"]),
    ]

    # Correlações internas da base Gen Z para o recorte selecionado.
    genz_correlation_columns = [
        "daily_usage_hours",
        "addiction_level_num",
        "mental_health_score",
        "num_platforms_used",
        "avg_session_minutes",
        "night_usage",
        "screen_time_before_sleep",
    ]
    normalized_genz = normalize_for_correlation(
        filtered,
        genz_correlation_columns,
    )
    usage_addiction_corr = normalized_genz["daily_usage_hours"].corr(
        normalized_genz["addiction_level_num"]
    )
    usage_mental_corr = normalized_genz["daily_usage_hours"].corr(
        normalized_genz["mental_health_score"]
    )
    platform_usage = (
        filtered.groupby("primary_platform")["daily_usage_hours"]
        .mean()
        .sort_values(ascending=False)
    )
    top_platform = platform_usage.index[0]
    top_platform_hours = platform_usage.iloc[0]
    insight = html.Div(
        [
            html.Strong("Horas de uso de um aplicativo: "),
            html.Span(
                f"{top_platform} é a plataforma com maior média de uso na seleção atual, "
                f"com {top_platform_hours:.2f} horas por dia."
            ),
        ]
    )

    # Dispersão baseada em amostra para manter a interação fluida.
    sample = filter_genz(countries, genders, None, source=genz_sample)
    fig_scatter = px.scatter(
        sample,
        x="daily_usage_hours",
        y="mental_health_score",
        color="primary_platform",
        size="avg_session_minutes",
        opacity=0.45,
        color_discrete_sequence=DISCRETE_SEQUENCE,
        title=f"Uso diário e saúde mental (r = {usage_mental_corr:.2f})",
        labels={
            "daily_usage_hours": "Uso diário (horas)",
            "mental_health_score": "Pontuação de saúde mental",
            "primary_platform": "Plataforma",
            "avg_session_minutes": "Duração média da sessão",
        },
    )
    fig_scatter = style_fig(fig_scatter)

    addiction_labels = {"Low": "Baixo", "Medium": "Médio", "High": "Alto"}
    box_data = filtered.assign(
        nivel_dependencia=filtered["addiction_level"].map(addiction_labels)
    )
    fig_addiction = px.box(
        box_data,
        x="nivel_dependencia",
        y="daily_usage_hours",
        color="nivel_dependencia",
        category_orders={"nivel_dependencia": ["Baixo", "Médio", "Alto"]},
        color_discrete_sequence=[COLORS["success"], COLORS["accent"], COLORS["danger"]],
        title=f"Uso diário por nível de dependência (r = {usage_addiction_corr:.2f})",
        labels={
            "nivel_dependencia": "Nível de dependência",
            "daily_usage_hours": "Uso diário (horas)",
        },
    )
    fig_addiction = style_fig(fig_addiction)

    # Relação entre bases: as médias são combinadas apenas depois da agregação
    # por país, gênero e plataforma, dimensões compartilhadas pelos dois CSVs.
    if filtered_teen.empty:
        fig_night_academic = empty_figure(
            "Uso noturno e desempenho acadêmico",
            "Adicione teen_behavior_patterns.csv para exibir esta relação.",
        )
    else:
        group_keys = ["country", "gender", "platform"]
        night_groups = (
            filtered.assign(
                platform=lambda frame: frame["primary_platform"].replace(
                    {"Twitter": "X/Twitter"}
                )
            )
            .groupby(group_keys, as_index=False)
            .agg(
                night_usage_rate=("night_usage", "mean"),
                genz_records=("night_usage", "size"),
            )
        )
        academic_groups = (
            filtered_teen.groupby(group_keys, as_index=False)
            .agg(
                academic_performance=(
                    "academic_performance_score",
                    "mean",
                ),
                teen_records=("academic_performance_score", "size"),
            )
        )
        night_academic = night_groups.merge(
            academic_groups,
            on=group_keys,
            how="inner",
        ).assign(
            night_usage_percent=lambda frame: frame["night_usage_rate"] * 100,
            gender_label=lambda frame: frame["gender"].map(TRANSLATIONS),
            records=lambda frame: frame[["genz_records", "teen_records"]].min(
                axis=1
            ),
        )

        if night_academic.empty:
            fig_night_academic = empty_figure(
                "Uso noturno e desempenho acadêmico",
                "Não há grupos em comum entre as bases para os filtros selecionados.",
            )
        else:
            night_academic_corr = night_academic["night_usage_percent"].corr(
                night_academic["academic_performance"]
            )
            fig_night_academic = px.scatter(
                night_academic,
                x="night_usage_percent",
                y="academic_performance",
                color="gender_label",
                symbol="gender_label",
                size="records",
                size_max=18,
                opacity=0.75,
                color_discrete_sequence=DISCRETE_SEQUENCE,
                hover_data={
                    "country": True,
                    "platform": True,
                    "genz_records": True,
                    "teen_records": True,
                    "records": False,
                    "gender_label": False,
                },
                title=(
                    "Uso noturno e desempenho acadêmico "
                    f"(r = {night_academic_corr:.2f})"
                ),
                labels={
                    "night_usage_percent": "Uso noturno médio (%)",
                    "academic_performance": "Desempenho acadêmico médio",
                    "gender_label": "Gênero",
                    "country": "País",
                    "platform": "Plataforma",
                    "genz_records": "Registros Gen Z",
                    "teen_records": "Registros adolescentes",
                },
            )

            # Linha de tendência linear calculada para todos os grupos visíveis.
            if (
                len(night_academic) >= 2
                and night_academic["night_usage_percent"].nunique() > 1
            ):
                slope, intercept = np.polyfit(
                    night_academic["night_usage_percent"],
                    night_academic["academic_performance"],
                    1,
                )
                trend_x = np.linspace(
                    night_academic["night_usage_percent"].min(),
                    night_academic["night_usage_percent"].max(),
                    100,
                )
                fig_night_academic.add_trace(
                    go.Scatter(
                        x=trend_x,
                        y=slope * trend_x + intercept,
                        mode="lines",
                        name="Tendência geral",
                        line=dict(color=COLORS["text"], dash="dash", width=2),
                        hoverinfo="skip",
                    )
                )
            fig_night_academic = style_fig(fig_night_academic)

    # Mapa completo das correlações entre as variáveis numéricas da base Gen Z.
    corr_columns = {
        "daily_usage_hours": "Uso diário",
        "addiction_level_num": "Nível de dependência",
        "mental_health_score": "Saúde mental",
        "num_platforms_used": "Nº de plataformas",
        "avg_session_minutes": "Duração da sessão",
        "night_usage": "Uso noturno",
        "screen_time_before_sleep": "Tela antes de dormir",
    }
    complete_corr = normalized_genz[list(corr_columns)].corr(numeric_only=True)
    complete_corr = complete_corr.rename(
        index=corr_columns,
        columns=corr_columns,
    )
    fig_strong = px.imshow(
        complete_corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=["#06B6D4", "#FFFFFF", "#4F46E5"],
        zmin=-1,
        zmax=1,
        title="Mapa completo de correlações de uso e bem-estar",
        labels={"color": "Correlação"},
    )
    fig_strong.update_xaxes(tickangle=-30)
    fig_strong = style_fig(fig_strong)

    # Média do indicador binário: ao multiplicar por 100, obtém-se a proporção
    # percentual de registros sinalizados em cada plataforma.
    if filtered_teen.empty or "suicide_risk_indicator" not in filtered_teen:
        fig_risk = empty_figure(
            "Indicador de risco de suicídio (%) por plataforma",
            "Adicione teen_behavior_patterns.csv para exibir este indicador.",
        )
    else:
        risk_data = (
            filtered_teen.groupby("platform", as_index=False)["suicide_risk_indicator"]
            .mean()
            .assign(risk_rate=lambda frame: frame["suicide_risk_indicator"] * 100)
            .sort_values("risk_rate", ascending=False)
        )
        fig_risk = px.bar(
            risk_data,
            x="platform",
            y="risk_rate",
            color="risk_rate",
            text="risk_rate",
            color_continuous_scale=["#DBEAFE", COLORS["danger"]],
            title="Indicador de risco de suicídio (%) por plataforma",
            labels={
                "platform": "Plataforma",
                "risk_rate": "Indicador de risco de suicídio (%)",
            },
        )
        fig_risk.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            cliponaxis=False,
        )
        fig_risk.update_layout(coloraxis_showscale=False)
        fig_risk = style_fig(fig_risk)

    # A ansiedade média da base de saúde mental é agregada pelas chaves comuns e
    # associada a uma amostra Gen Z para contextualizar uso e dependência.
    if filtered_mental.empty:
        fig_usage_addiction_mental = empty_figure(
            "Uso, dependência e indicadores de saúde mental"
        )
    else:
        mental_metrics = {"anxiety_score": "Ansiedade"}
        genz_records = (
            filtered.sample(
                n=min(4000, len(filtered)),
                random_state=42,
            )
            .assign(
                platform=lambda frame: frame["primary_platform"].replace(
                    {"Twitter": "X/Twitter"}
                )
            )[
                [
                    "country",
                    "gender",
                    "platform",
                    "daily_usage_hours",
                    "addiction_level_num",
                ]
            ]
        )
        mental_groups = (
            filtered_mental.groupby(
                ["country", "gender", "platform"],
                as_index=False,
            )[list(mental_metrics)]
            .mean()
        )
        combined_records = genz_records.merge(
            mental_groups,
            on=["country", "gender", "platform"],
            how="inner",
        )
        long_mental = (
            combined_records
            .melt(
                id_vars=[
                    "country",
                    "gender",
                    "platform",
                    "daily_usage_hours",
                    "addiction_level_num",
                ],
                value_vars=list(mental_metrics),
                var_name="metric",
                value_name="raw_value",
            )
            .assign(
                indicator=lambda frame: frame["metric"].map(mental_metrics),
                normalized_value=lambda frame: np.where(
                    frame["metric"].eq("mental_health_risk_num"),
                    frame["raw_value"] / 3 * 100,
                    frame["raw_value"],
                ),
            )
        )
        fig_usage_addiction_mental = px.scatter(
            long_mental,
            x="daily_usage_hours",
            y="addiction_level_num",
            color="raw_value",
            size="normalized_value",
            hover_data={
                "country": True,
                "gender": True,
                "platform": True,
                "raw_value": ":.2f",
                "normalized_value": False,
                "addiction_level_num": False,
            },
            color_continuous_scale=["#DBEAFE", COLORS["danger"]],
            size_max=18,
            opacity=0.72,
            title="Uso diário, dependência e ansiedade",
            labels={
                "daily_usage_hours": "Horas de uso diário",
                "addiction_level_num": "Nível de dependência",
                "raw_value": "Ansiedade média",
                "country": "País",
                "gender": "Gênero",
                "platform": "Plataforma",
            },
        )
        fig_usage_addiction_mental.update_yaxes(
            tickmode="array",
            tickvals=[1, 2, 3],
            ticktext=["Baixo", "Médio", "Alto"],
            range=[0.7, 3.3],
        )
        fig_usage_addiction_mental = style_fig(fig_usage_addiction_mental)
        fig_usage_addiction_mental.update_layout(
            margin=dict(l=40, r=30, t=70, b=100)
        )

    return (
        kpis,
        insight,
        fig_scatter,
        fig_addiction,
        fig_night_academic,
        fig_strong,
        fig_risk,
        fig_usage_addiction_mental,
    )


if __name__ == "__main__":
    # O modo debug recarrega o servidor automaticamente durante o desenvolvimento.
    app.run(debug=True)
