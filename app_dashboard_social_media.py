# ============================================================
# DASHBOARD: Social Media, Well-being and Teen Behavior
# Arquivo principal: app_dashboard_social_media.py
#
# VISÃO GERAL DO FUNCIONAMENTO:
# 1) O pandas lê e prepara os dois arquivos CSV.
# 2) app.layout cria a interface visual usando componentes do Dash.
# 3) O usuário altera filtros, botões ou o intervalo de idade.
# 4) O Dash envia os novos valores para update_dashboard().
# 5) A função filtra os dados, calcula indicadores e cria os gráficos.
# 6) Os valores retornados atualizam automaticamente os componentes da página.
#
# Como rodar:
# 1) Instale as bibliotecas:
#    pip install dash plotly pandas numpy
#
# 2) Deixe os CSVs na mesma pasta deste arquivo:
#    - genz_social_media_usage_1M (1).csv
#    - teen_behavior_patterns.csv
#
# 3) Execute:
#    python app_dashboard_social_media.py
#
# 4) Abra no navegador:
#    http://127.0.0.1:8050
# ============================================================

from pathlib import Path  # Cria caminhos de arquivos de forma portável.

# NumPy trabalha bem com vetores e colunas inteiras, sem usar laços linha por linha.
import numpy as np  # Aqui é usado principalmente em np.where.

# Um DataFrame pode ser entendido como uma tabela com linhas e colunas nomeadas.
import pandas as pd  # Carrega, filtra e resume os dados tabulares.

# Plotly Express oferece funções prontas como bar, scatter, box e histogram.
import plotly.express as px  # Cria gráficos Plotly com uma API de alto nível.

# graph_objects oferece controle mais manual; aqui cria a figura sem dados.
import plotly.graph_objects as go  # Cria figuras Plotly manualmente.

# Dash transforma componentes Python em HTML/JavaScript executado no navegador.
# dcc contém componentes interativos; html contém equivalentes das tags HTML.
# Input informa o que dispara um callback; Output informa o que ele atualiza.
from dash import Dash, Input, Output, dcc, html  # Monta a interface e seus callbacks.

# ------------------------------------------------------------
# 1. Carregamento dos dados
# ------------------------------------------------------------
# __file__ é este arquivo; resolve() gera o caminho absoluto e parent obtém sua pasta.
BASE_DIR = Path(__file__).resolve().parent

# O operador / do Path combina a pasta-base com o nome de cada CSV.
GENZ_PATH = BASE_DIR / "genz_social_media_usage_1M (1).csv"
TEEN_PATH = BASE_DIR / "teen_behavior_patterns.csv"

# read_csv interpreta a primeira linha como nomes das colunas e cada linha seguinte
# como um registro. O resultado é carregado em memória como um DataFrame.
genz = pd.read_csv(GENZ_PATH)
teen = pd.read_csv(TEEN_PATH)

# np.where cria um rótulo legível para cada valor binário de night_usage.
# Para cada linha: se night_usage == 1, usa o primeiro texto; senão, usa o segundo.
genz["night_usage_label"] = np.where(
    genz["night_usage"] == 1,
    "Uses at night",
    "Does not use at night",
)

# len(genz) informa o número de linhas e min evita solicitar uma amostra maior que a base.
# sample sorteia linhas. Limitar o scatter evita enviar 1 milhão de pontos ao navegador.
# random_state=42 fixa a semente aleatória, mantendo a mesma amostra em cada execução.
genz_sample = genz.sample(n=min(35000, len(genz)), random_state=42)

# ------------------------------------------------------------
# 2. Paleta visual
# Tema: tecnologia + redes sociais + saúde mental
# ------------------------------------------------------------
COLORS = {
    # Fundo geral da página.
    "background": "#F6F8FB",
    # Fundo branco usado em cards e gráficos.
    "card": "#FFFFFF",
    # Cor principal dos textos.
    "text": "#1F2937",
    # Cor menos intensa para textos secundários.
    "muted": "#6B7280",
    "primary": "#4F46E5",   # índigo
    "secondary": "#06B6D4", # ciano
    "accent": "#F97316",    # laranja
    "danger": "#E11D48",    # rosa/vermelho
    "success": "#10B981",
    "border": "#E5E7EB",
}

# Template visual básico aplicado a todos os gráficos.
PLOTLY_TEMPLATE = "plotly_white"

# Cores usadas em sequência quando um gráfico possui várias categorias.
DISCRETE_SEQUENCE = ["#4F46E5", "#06B6D4", "#F97316", "#10B981", "#E11D48", "#8B5CF6"]

# ------------------------------------------------------------
# 3. Funções auxiliares
# ------------------------------------------------------------
def options_from(series):
    """Converte valores únicos de uma Series em opções aceitas por Dropdown."""
    # dropna ignora ausentes, unique remove duplicatas e sorted ordena os valores.
    values = sorted(series.dropna().unique())
    # A compreensão de lista executa o dicionário para cada valor encontrado.
    # label é o texto que o usuário vê; value é o dado enviado ao callback.
    return [{"label": str(v), "value": v} for v in values]


def filter_genz(
    selected_countries,
    selected_genders,
    selected_platforms,
    age_range,
    data=None,
):
    """Aplica à base Gen Z somente os filtros que possuem uma seleção."""
    # Os quatro primeiros parâmetros recebem os valores dos componentes do layout.
    # data é opcional: permite filtrar tanto a base completa quanto uma amostra.
    # Usa a base completa por padrão, mas também aceita a amostra do scatter.
    # Não é necessário copy(): cada filtro abaixo já produz um novo DataFrame.
    df = genz if data is None else data

    # Uma lista vazia ou None é avaliada como False; nesse caso, o filtro é ignorado.
    if selected_countries:
        # isin mantém somente linhas cujo país aparece na lista selecionada.
        # A expressão entre colchetes gera uma Series de True/False para cada linha.
        df = df[df["country"].isin(selected_countries)]

    if selected_genders:
        # Somente linhas marcadas como True pela máscara permanecem em df.
        df = df[df["gender"].isin(selected_genders)]

    if selected_platforms:
        df = df[df["primary_platform"].isin(selected_platforms)]

    if age_range:
        # & combina os limites inferior e superior em uma única máscara booleana.
        # Os parênteses são necessários porque cada comparação é feita separadamente.
        df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

    # A função devolve a versão filtrada sem modificar a variável global genz.
    return df


def filter_teen(selected_countries, selected_genders, selected_platforms):
    """Aplica filtros à base teen, que usa a coluna platform em vez de primary_platform."""
    # A lógica é igual à de filter_genz, mas esta base não recebe filtro de idade.
    df = teen

    if selected_countries:
        df = df[df["country"].isin(selected_countries)]

    if selected_genders:
        df = df[df["gender"].isin(selected_genders)]

    if selected_platforms:
        df = df[df["platform"].isin(selected_platforms)]

    return df


def empty_figure(title):
    """Cria um gráfico vazio com uma mensagem quando os filtros não retornam dados."""
    # go.Figure inicia uma figura sem séries.
    fig = go.Figure()
    # update_layout define apenas a aparência e a anotação central.
    fig.update_layout(
        # Título recebido como argumento pela função.
        title=title,
        # Tema visual padrão definido no início do arquivo.
        template=PLOTLY_TEMPLATE,
        # paper_bgcolor é o fundo externo; plot_bgcolor é a área dos eixos.
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        # annotations permite escrever texto livre dentro do gráfico.
        annotations=[
            dict(
                text="No data available for the selected filters.",
                # x=0.5 e y=0.5 posicionam o texto no centro.
                x=0.5,
                y=0.5,
                # Não desenha uma seta apontando para o texto.
                showarrow=False,
                font=dict(size=16, color=COLORS["muted"]),
            )
        ],
        # Como não há dados, os dois eixos são ocultados.
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def style_fig(fig):
    """Aplica a mesma identidade visual a qualquer figura Plotly."""
    # As chamadas update_* alteram a própria figura recebida.
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        # Define a tipografia de todos os textos do gráfico.
        font=dict(color=COLORS["text"], family="Arial"),
        # x=0.02 alinha o título próximo à margem esquerda.
        title=dict(font=dict(size=18), x=0.02),
        # Margens: esquerda, direita, topo e base.
        margin=dict(l=35, r=25, t=65, b=45),
        # Remove o título automático exibido sobre a legenda.
        legend_title_text="",
    )
    # Remove linhas de grade verticais e suaviza as horizontais.
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF2F7")
    return fig


def kpi_card(title, value, subtitle, color):
    """Monta um card reutilizável com título, valor principal e subtítulo."""
    # html.Div é um contêiner; children define seus elementos internos.
    return html.Div(
        children=[
            # html.P gera um parágrafo pequeno com o nome do indicador.
            html.P(title, style={"margin": "0", "fontSize": "13px", "color": COLORS["muted"]}),
            # html.H2 destaca o valor calculado usando a cor recebida pela função.
            html.H2(value, style={"margin": "8px 0 4px", "fontSize": "28px", "color": color}),
            # O segundo parágrafo explica o significado ou a origem do valor.
            html.P(subtitle, style={"margin": "0", "fontSize": "12px", "color": COLORS["muted"]}),
        ],
        style={
            # Este dicionário equivale a regras CSS aplicadas diretamente ao elemento.
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            # borderRadius arredonda os cantos.
            "borderRadius": "16px",
            # padding cria espaço interno entre o conteúdo e as bordas.
            "padding": "18px",
            # boxShadow cria profundidade visual por meio de uma sombra.
            "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.06)",
            "minHeight": "110px",
        },
    )

# ------------------------------------------------------------
# 4. Aplicação Dash
# ------------------------------------------------------------
# Cria o servidor Dash; __name__ ajuda o framework a localizar recursos do projeto.
app = Dash(__name__)
# Define o texto mostrado na aba do navegador.
app.title = "Social Media & Teen Behavior Dashboard"

# layout descreve toda a árvore de componentes HTML exibida na página.
# No Dash, a interface é construída como uma árvore: Divs externas contêm Divs,
# textos, controles e gráficos internos por meio da propriedade children.
app.layout = html.Div(
    # O style externo define o fundo, espaçamento e tipografia da página inteira.
    style={
        "backgroundColor": COLORS["background"],
        # 100vh significa 100% da altura visível da janela do navegador.
        "minHeight": "100vh",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
        "color": COLORS["text"],
    },
    children=[
        # Cabeçalho do dashboard.
        html.Div(
            children=[
                # H1 representa o título principal da página.
                html.H1(
                    "Social Media, Well-being and Teen Behavior",
                    style={"margin": "0", "fontSize": "34px", "color": COLORS["text"]},
                ),
                # P representa um parágrafo explicativo abaixo do título.
                html.P(
                    "An interactive dashboard to understand how platform choice, usage intensity and social pressure indicators relate to well-being patterns among Gen Z and teenagers.",
                    style={"marginTop": "8px", "maxWidth": "980px", "color": COLORS["muted"], "fontSize": "16px"},
                ),
            ],
            style={"marginBottom": "22px"},
        ),

        # -------------------------
        # Filtros
        # -------------------------
        html.Div(
            children=[
                # Dropdown multi-seleção para países presentes em qualquer uma das bases.
                html.Div(
                    children=[
                        html.Label("Countries", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            # id identifica o componente e permite referenciá-lo no callback.
                            id="country-filter",
                            # concat une as duas colunas antes de gerar opções únicas.
                            options=options_from(pd.concat([genz["country"], teen["country"]])),
                            # value define a seleção inicial exibida ao abrir o dashboard.
                            value=["Brazil", "USA", "UK"],
                            # multi=True permite selecionar mais de uma opção.
                            multi=True,
                            # placeholder aparece quando nenhuma opção está selecionada.
                            placeholder="Select countries",
                        ),
                    ],
                    style={"flex": "2", "minWidth": "260px"},
                ),
                # Dropdown de gênero; value=None significa que nenhum filtro é aplicado.
                html.Div(
                    children=[
                        html.Label("Gender", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="gender-filter",
                            options=options_from(pd.concat([genz["gender"], teen["gender"]])),
                            # None representa ausência de seleção e, portanto, todos os gêneros.
                            value=None,
                            multi=True,
                            placeholder="All genders",
                        ),
                    ],
                    style={"flex": "1.2", "minWidth": "220px"},
                ),
                # As plataformas vêm de colunas com nomes diferentes nas duas bases.
                html.Div(
                    children=[
                        html.Label("Platforms", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="platform-filter",
                            options=options_from(pd.concat([genz["primary_platform"], teen["platform"]])),
                            value=None,
                            multi=True,
                            placeholder="All platforms",
                        ),
                    ],
                    style={"flex": "2", "minWidth": "260px"},
                ),
                # RangeSlider devolve uma lista [idade_mínima, idade_máxima].
                html.Div(
                    children=[
                        html.Label("Gen Z age range", style={"fontWeight": "bold"}),
                        dcc.RangeSlider(
                            id="age-filter",
                            # Os limites são calculados diretamente dos dados.
                            min=int(genz["age"].min()),
                            max=int(genz["age"].max()),
                            value=[int(genz["age"].min()), int(genz["age"].max())],
                            # marks define os valores escritos abaixo da barra deslizante.
                            marks={13: "13", 18: "18", 23: "23", 27: "27"},
                            # tooltip mostra o número atual enquanto o controle é movimentado.
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    style={"flex": "1.8", "minWidth": "260px", "padding": "0 8px"},
                ),
            ],
            # Flexbox mantém os filtros lado a lado e permite quebra em telas menores.
            style={
                # display flex organiza os filhos em uma linha flexível.
                "display": "flex",
                # gap controla a distância entre os componentes.
                "gap": "16px",
                # flexWrap permite mover componentes para a linha seguinte.
                "flexWrap": "wrap",
                "backgroundColor": COLORS["card"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "18px",
                "padding": "18px",
                "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.05)",
                "marginBottom": "18px",
            },
        ),

        # -------------------------
        # Botões interativos
        # -------------------------
        html.Div(
            children=[
                # Define a dimensão categórica usada no eixo x do gráfico principal.
                html.Div(
                    children=[
                        html.Label("Compare Gen Z by", style={"fontWeight": "bold"}),
                        dcc.RadioItems(
                            id="compare-by",
                            # Cada value corresponde exatamente ao nome de uma coluna de genz.
                            options=[
                                {"label": "Platform", "value": "primary_platform"},
                                {"label": "Country", "value": "country"},
                                {"label": "Gender", "value": "gender"},
                                {"label": "Purpose", "value": "purpose"},
                            ],
                            # Apenas uma opção de RadioItems pode ficar marcada.
                            value="primary_platform",
                            # inline=True coloca as opções lado a lado.
                            inline=True,
                            # inputStyle estiliza o círculo do botão.
                            inputStyle={"marginRight": "6px"},
                            # labelStyle estiliza a área de texto ao redor de cada botão.
                            labelStyle={
                                "marginRight": "16px",
                                "padding": "8px 12px",
                                "border": f"1px solid {COLORS['border']}",
                                "borderRadius": "999px",
                                "backgroundColor": "#F9FAFB",
                            },
                        ),
                    ],
                    style={"flex": "2"},
                ),
                # Define qual coluna numérica será resumida no gráfico principal.
                html.Div(
                    children=[
                        html.Label("Main metric", style={"fontWeight": "bold"}),
                        dcc.RadioItems(
                            id="metric-choice",
                            # Os values são nomes de colunas numéricas usadas no cálculo da média.
                            options=[
                                {"label": "Daily usage", "value": "daily_usage_hours"},
                                {"label": "Mental health", "value": "mental_health_score"},
                                {"label": "Before sleep", "value": "screen_time_before_sleep"},
                                {"label": "Session length", "value": "avg_session_minutes"},
                            ],
                            value="daily_usage_hours",
                            inline=True,
                            inputStyle={"marginRight": "6px"},
                            labelStyle={
                                "marginRight": "16px",
                                "padding": "8px 12px",
                                "border": f"1px solid {COLORS['border']}",
                                "borderRadius": "999px",
                                "backgroundColor": "#F9FAFB",
                            },
                        ),
                    ],
                    style={"flex": "3"},
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

        # -------------------------
        # KPIs
        # -------------------------
        # O callback preencherá children com os cinco cards calculados.
        html.Div(
            id="kpi-row",
            style={
                # CSS Grid organiza automaticamente os cards em colunas.
                "display": "grid",
                # auto-fit cria quantas colunas couberem; cada uma terá no mínimo 210 px.
                "gridTemplateColumns": "repeat(auto-fit, minmax(210px, 1fr))",
                "gap": "16px",
                "marginBottom": "18px",
            },
        ),

        # -------------------------
        # Narrativa
        # -------------------------
        # O callback preencherá este contêiner com um insight textual automático.
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

        # -------------------------
        # Visualizações
        # -------------------------
        html.Div(
            children=[
                # Cada dcc.Graph recebe sua propriedade figure por meio do callback.
                # span 6 ocupa metade da grade de 12 colunas.
                html.Div(dcc.Graph(id="bar-main-metric"), className="chart-card", style={"gridColumn": "span 6"}),
                html.Div(dcc.Graph(id="scatter-usage-mental"), className="chart-card", style={"gridColumn": "span 6"}),
                # span 4 ocupa um terço da grade.
                html.Div(dcc.Graph(id="box-addiction"), className="chart-card", style={"gridColumn": "span 4"}),
                html.Div(dcc.Graph(id="hist-before-sleep"), className="chart-card", style={"gridColumn": "span 4"}),
                html.Div(dcc.Graph(id="teen-risk-platform"), className="chart-card", style={"gridColumn": "span 4"}),
                # span 8 ocupa dois terços da grade.
                html.Div(dcc.Graph(id="teen-year-line"), className="chart-card", style={"gridColumn": "span 8"}),
                html.Div(dcc.Graph(id="correlation-heatmap"), className="chart-card", style={"gridColumn": "span 4"}),
            ],
            # A grade tem 12 colunas; cada card informa quantas deve ocupar.
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(12, 1fr)",
                "gap": "16px",
            },
        ),

        html.Div(
            # Bloco estático: seu conteúdo não é alterado pelo callback.
            children=[
                html.H3("How to read this dashboard", style={"marginBottom": "8px"}),
                html.P(
                    "Start with the KPI cards, then compare platforms/countries/genders using the button group. "
                    "The strongest story is not only which platform has more usage, but whether high usage appears together with lower mental health scores, night usage, pressure indicators or risk signals.",
                    style={"color": COLORS["muted"], "lineHeight": "1.5"},
                ),
            ],
            style={
                "marginTop": "18px",
                "backgroundColor": COLORS["card"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "18px",
                "padding": "18px",
            },
        ),
    ],
)

# index_string substitui o HTML-base padrão do Dash para incluir CSS adicional.
# A letra f antes das aspas permite inserir valores Python, como COLORS["card"],
# dentro da string usando chaves. As chaves duplas preservam chaves literais no CSS.
app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        <!-- As expressões entre chaves são marcadores que o Dash substitui. -->
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            /* Aparência compartilhada por todos os contêineres de gráficos. */
            .chart-card {{
                background-color: {COLORS["card"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 18px;
                padding: 8px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
                /* min-width: 0 impede que gráficos largos estourem a coluna da grade. */
                min-width: 0;
            }}
            /* Em telas estreitas, cada gráfico passa a ocupar a linha inteira. */
            @media (max-width: 1100px) {{
                .chart-card {{
                    grid-column: span 12 !important;
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

# ------------------------------------------------------------
# 5. Callbacks
# ------------------------------------------------------------
# O decorador registra a função abaixo como reação aos seis Inputs.
# A ordem dos nove valores retornados deve corresponder à ordem dos Outputs.
@app.callback(
    # Cada Output contém: id do componente que será alterado e sua propriedade.
    # "children" recebe componentes/texto; "figure" recebe uma figura Plotly.
    Output("kpi-row", "children"),
    Output("insight-box", "children"),
    Output("bar-main-metric", "figure"),
    Output("scatter-usage-mental", "figure"),
    Output("box-addiction", "figure"),
    Output("hist-before-sleep", "figure"),
    Output("teen-risk-platform", "figure"),
    Output("teen-year-line", "figure"),
    Output("correlation-heatmap", "figure"),
    # Cada Input observa a propriedade value de um controle da interface.
    # Quando qualquer value muda, update_dashboard é executada novamente.
    Input("country-filter", "value"),
    Input("gender-filter", "value"),
    Input("platform-filter", "value"),
    Input("age-filter", "value"),
    Input("compare-by", "value"),
    Input("metric-choice", "value"),
)
def update_dashboard(selected_countries, selected_genders, selected_platforms, age_range, compare_by, metric_choice):
    """Recalcula cards, insight e gráficos sempre que um filtro muda.

    A ordem dos parâmetros segue a ordem dos Inputs declarados no decorador.
    A ordem dos itens retornados segue a ordem dos Outputs.
    """
    # Filtra cada base com os valores atuais dos componentes da interface.
    df_g = filter_genz(selected_countries, selected_genders, selected_platforms, age_range)
    df_t = filter_teen(selected_countries, selected_genders, selected_platforms)

    # Sem dados Gen Z, não há métricas principais; devolve figuras informativas vazias.
    if df_g.empty:
        # A tupla possui nove posições porque o callback declarou nove Outputs.
        # Mesmo sem dados, todos os Outputs precisam receber algum valor.
        return (
            [kpi_card("No data", "-", "Change the filters", COLORS["danger"])],
            "There is no Gen Z data for the selected filters.",
            empty_figure("Main metric comparison"),
            empty_figure("Daily usage vs mental health"),
            empty_figure("Mental health by addiction level"),
            empty_figure("Screen time before sleep"),
            empty_figure("Teen risk by platform"),
            empty_figure("Teen indicators over time"),
            empty_figure("Correlation heatmap"),
        )

    # KPIs
    # mean calcula a média da coluna; para night_usage, 0/1 vira uma proporção.
    # Valores ausentes (NaN) são ignorados por mean por padrão.
    avg_usage = df_g["daily_usage_hours"].mean()
    avg_mental = df_g["mental_health_score"].mean()
    night_share = df_g["night_usage"].mean() * 100
    avg_sleep = df_g["screen_time_before_sleep"].mean()

    # A base teen pode ficar vazia mesmo quando a base Gen Z contém dados.
    if df_t.empty:
        risk_text = "-"
        risk_sub = "Teen data unavailable under current filters"
    else:
        # Booleanos são tratados como 0/1; multiplicar por 100 converte em percentual.
        risk_rate = df_t["suicide_risk_indicator"].mean() * 100
        risk_text = f"{risk_rate:.1f}%"
        risk_sub = "Teen records with risk indicator"

    # Cada chamada cria um componente Dash; as f-strings formatam casas decimais e unidades.
    # :.2f mantém duas casas decimais e :.1f mantém uma casa decimal.
    kpis = [
        kpi_card("Avg. daily usage", f"{avg_usage:.2f} h", "Gen Z social media use", COLORS["primary"]),
        # A expressão condicional escolhe verde se a média for pelo menos 7.
        kpi_card("Avg. mental health score", f"{avg_mental:.2f}/10", "Higher score means better indicator", COLORS["success"] if avg_mental >= 7 else COLORS["accent"]),
        kpi_card("Night usage share", f"{night_share:.1f}%", "Users active at night", COLORS["accent"]),
        kpi_card("Before-sleep screen time", f"{avg_sleep:.1f} min", "Average screen time before sleeping", COLORS["secondary"]),
        kpi_card("Teen risk indicator", risk_text, risk_sub, COLORS["danger"]),
    ]

    # Insight narrativo automático
    # Agrupa por plataforma, calcula a média, ordena e conserva somente a maior.
    # groupby divide o DataFrame em grupos com o mesmo valor de primary_platform.
    top_platform = (
        df_g.groupby("primary_platform")["daily_usage_hours"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
    )
    # index[0] recupera o nome da plataforma; iloc[0] recupera sua média.
    top_platform_name = top_platform.index[0]
    top_platform_value = top_platform.iloc[0]

    insight = html.Div(
        # O texto é dividido em Strong (negrito) e Span (texto normal).
        [
            html.Strong("Key story: "),
            html.Span(
                f"In the current selection, {top_platform_name} has the highest average daily usage "
                f"({top_platform_value:.2f} hours/day). Compare this with the scatter plot and addiction-level boxplot "
                f"to see whether heavier use appears alongside lower well-being indicators or more night usage."
            ),
        ]
    )

    # 1. Bar chart: main metric comparison
    metric_labels = {
        # Este dicionário traduz nomes técnicos de colunas em títulos legíveis.
        "daily_usage_hours": "Average daily usage (hours)",
        "mental_health_score": "Average mental health score",
        "screen_time_before_sleep": "Average screen time before sleep (minutes)",
        "avg_session_minutes": "Average session length (minutes)",
    }

    # as_index=False mantém compare_by como coluna comum, adequada ao Plotly Express.
    bar_df = (
        # [metric_choice] seleciona dinamicamente a coluna escolhida pelo usuário.
        df_g.groupby(compare_by, as_index=False)[metric_choice]
        # Calcula uma média para cada categoria de compare_by.
        .mean()
        # Coloca as maiores médias no início do DataFrame.
        .sort_values(metric_choice, ascending=False)
    )

    # px.bar mapeia colunas do DataFrame para eixos, cor, texto e rótulos.
    fig_bar = px.bar(
        bar_df,
        x=compare_by,
        y=metric_choice,
        # color cria uma cor diferente para cada categoria.
        color=compare_by,
        # text informa qual coluna será escrita sobre as barras.
        text=metric_choice,
        color_discrete_sequence=DISCRETE_SEQUENCE,
        title=f"{metric_labels[metric_choice]} by {compare_by.replace('_', ' ').title()}",
        labels={compare_by: compare_by.replace("_", " ").title(), metric_choice: metric_labels[metric_choice]},
    )
    # Mostra o valor médio sobre cada barra.
    # %{text:.2f} é a sintaxe do Plotly para mostrar duas casas decimais.
    # cliponaxis=False permite que o texto apareça fora da área do eixo.
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig_bar = style_fig(fig_bar)

    # 2. Scatter: usage vs mental health
    # Reaproveita a função de filtros sobre a amostra para evitar código duplicado.
    df_sample = filter_genz(
        selected_countries,
        selected_genders,
        selected_platforms,
        age_range,
        data=genz_sample,
    )

    if df_sample.empty:
        fig_scatter = empty_figure("Daily usage vs mental health")
    else:
        # Cada ponto é um registro; tamanho representa duração e cor representa plataforma.
        # O scatter ajuda a observar associação entre duas variáveis, mas não prova causalidade.
        fig_scatter = px.scatter(
            df_sample,
            # Posição horizontal de cada pessoa.
            x="daily_usage_hours",
            # Posição vertical da mesma pessoa.
            y="mental_health_score",
            # Separa visualmente as plataformas.
            color="primary_platform",
            # Sessões mais longas produzem círculos maiores.
            size="avg_session_minutes",
            # Transparência reduz a sobreposição visual entre muitos pontos.
            opacity=0.45,
            color_discrete_sequence=DISCRETE_SEQUENCE,
            title="Does heavier daily usage appear with different mental health scores?",
            labels={
                "daily_usage_hours": "Daily usage (hours)",
                "mental_health_score": "Mental health score",
                "primary_platform": "Platform",
                "avg_session_minutes": "Avg. session minutes",
            },
        )
        fig_scatter = style_fig(fig_scatter)

    # 3. Boxplot: mental health by addiction level
    # category_orders força uma ordem semântica em vez da ordem alfabética.
    # O boxplot resume a distribuição: mediana, quartis, dispersão e possíveis outliers.
    order = ["Low", "Medium", "High"]
    fig_box = px.box(
        df_g,
        # Cada categoria forma uma caixa no eixo horizontal.
        x="addiction_level",
        # Os valores desta coluna formam a distribuição vertical.
        y="mental_health_score",
        # Aplica uma cor específica para cada nível.
        color="addiction_level",
        category_orders={"addiction_level": order},
        color_discrete_sequence=[COLORS["success"], COLORS["accent"], COLORS["danger"]],
        title="Mental health score by addiction level",
        labels={"addiction_level": "Addiction level", "mental_health_score": "Mental health score"},
    )
    fig_box = style_fig(fig_box)

    # 4. Histogram: screen time before sleep
    # overlay sobrepõe as distribuições dos dois grupos e opacity permite compará-las.
    # Um histograma conta quantos registros caem em cada intervalo numérico.
    fig_hist = px.histogram(
        df_g,
        # Variável contínua dividida em intervalos.
        x="screen_time_before_sleep",
        # Cria uma distribuição para usuários noturnos e outra para os demais.
        color="night_usage_label",
        # Solicita aproximadamente 35 intervalos (bins).
        nbins=35,
        # overlay desenha os grupos na mesma área, em vez de empilhá-los.
        barmode="overlay",
        opacity=0.72,
        color_discrete_sequence=[COLORS["secondary"], COLORS["accent"]],
        title="Distribution of screen time before sleep",
        labels={"screen_time_before_sleep": "Minutes before sleep", "night_usage_label": "Night usage"},
    )
    fig_hist = style_fig(fig_hist)

    # 5. Teen risk rate by platform
    if df_t.empty:
        # Evita executar groupby e construir um gráfico com uma tabela vazia.
        fig_risk = empty_figure("Teen risk indicator by platform")
    else:
        # assign cria risk_rate a partir da média booleana de cada plataforma.
        # Exemplo: média 0,25 significa que 25% dos registros têm indicador True/1.
        risk_df = (
            df_t.groupby("platform", as_index=False)["suicide_risk_indicator"]
            .mean()
            # lambda d recebe o DataFrame intermediário produzido pelo encadeamento.
            .assign(risk_rate=lambda d: d["suicide_risk_indicator"] * 100)
            .sort_values("risk_rate", ascending=False)
        )
        # Uma escala contínua colore barras de acordo com seu próprio percentual.
        fig_risk = px.bar(
            risk_df,
            x="platform",
            y="risk_rate",
            # A própria taxa controla a intensidade da cor de cada barra.
            color="risk_rate",
            color_continuous_scale=["#DBEAFE", COLORS["danger"]],
            text="risk_rate",
            title="Teen risk indicator rate by platform",
            labels={"platform": "Platform", "risk_rate": "Risk indicator rate (%)"},
        )
        # Inclui o símbolo de porcentagem no texto mostrado sobre as barras.
        fig_risk.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        # Esconde a barra lateral da escala de cores para deixar o gráfico mais limpo.
        fig_risk.update_layout(coloraxis_showscale=False)
        fig_risk = style_fig(fig_risk)

    # 6. Line chart: teen indicators over year
    if df_t.empty:
        fig_line = empty_figure("Teen indicators over time")
    else:
        # Calcula a média anual das quatro colunas de interesse.
        # O resultado inicial possui uma linha por ano e uma coluna por indicador.
        line_df = (
            df_t.groupby("year", as_index=False)[
                [
                    "academic_performance_score",
                    "social_comparison_index",
                    "body_image_anxiety_score",
                    "peer_pressure_score",
                ]
            ]
            .mean()
        )
        # melt transforma quatro colunas em pares indicador/score para desenhar quatro linhas.
        # Formato largo:
        # year | academic_performance | social_comparison | ...
        # Formato longo após melt:
        # year | indicator            | score
        line_long = line_df.melt(
            # year permanece como identificador e não é transformado.
            id_vars="year",
            # Nome da nova coluna que guardará os antigos nomes de colunas.
            var_name="indicator",
            # Nome da nova coluna que guardará os valores numéricos.
            value_name="score",
        )
        label_map = {
            "academic_performance_score": "Academic performance",
            "social_comparison_index": "Social comparison",
            "body_image_anxiety_score": "Body image anxiety",
            "peer_pressure_score": "Peer pressure",
        }
        # map troca nomes técnicos por rótulos legíveis.
        line_long["indicator"] = line_long["indicator"].map(label_map)

        fig_line = px.line(
            line_long,
            # Os anos aparecem no eixo horizontal em ordem temporal.
            x="year",
            # A média de cada indicador aparece no eixo vertical.
            y="score",
            # Cada indicador recebe uma linha e uma cor.
            color="indicator",
            # markers=True desenha pontos nas observações anuais.
            markers=True,
            color_discrete_sequence=DISCRETE_SEQUENCE,
            title="Teen behavior indicators over time",
            labels={"year": "Year", "score": "Average score", "indicator": "Indicator"},
        )
        fig_line = style_fig(fig_line)

    # 7. Correlation heatmap
    # Somente variáveis numéricas relevantes são incluídas na análise.
    corr_cols = [
        "daily_usage_hours",
        "num_platforms_used",
        "avg_session_minutes",
        "night_usage",
        "mental_health_score",
        "screen_time_before_sleep",
    ]
    # corr produz a matriz de correlação linear entre todas as colunas listadas.
    # O coeficiente varia de -1 a 1:
    # próximo de 1  = as variáveis tendem a crescer juntas;
    # próximo de -1 = quando uma cresce, a outra tende a diminuir;
    # próximo de 0  = pouca relação linear.
    # Correlação indica associação estatística, não uma relação de causa e efeito.
    corr = df_g[corr_cols].corr(numeric_only=True)
    # imshow representa a matriz como mapa de calor e escreve cada coeficiente na célula.
    fig_corr = px.imshow(
        corr,
        # Mostra o coeficiente com duas casas decimais dentro de cada célula.
        text_auto=".2f",
        # Faz o gráfico usar automaticamente o espaço disponível.
        aspect="auto",
        # Define as cores usadas do menor ao maior valor.
        color_continuous_scale=["#E0F2FE", "#FFFFFF", "#4F46E5"],
        title="Correlation between Gen Z usage and well-being variables",
        labels={"color": "Correlation"},
    )
    fig_corr = style_fig(fig_corr)

    # Os valores seguem exatamente a ordem declarada no decorador @app.callback.
    # O Dash converte as figuras e componentes Python para dados que o navegador renderiza.
    return kpis, insight, fig_bar, fig_scatter, fig_box, fig_hist, fig_risk, fig_line, fig_corr


# Impede que o servidor seja iniciado quando este arquivo for apenas importado.
if __name__ == "__main__":
    # Inicia o servidor local; debug recarrega o app após alterações e exibe erros detalhados.
    # Em produção, normalmente debug seria False por segurança e desempenho.
    app.run(debug=True)
