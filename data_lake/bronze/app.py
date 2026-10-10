import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data_lake" / "gold"
ZONE_PARQUET = PROJECT_ROOT / "geo_data" / "taxi_zones" / "taxi_zones.parquet"

# 1. Configuração do Ecrã do Dashboard
st.set_page_config(page_title="Dashboard NYC Taxi", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        color-scheme: light;
        --ink: #25322f;
        --muted: #687571;
        --paper: #f5f6f2;
        --line: #e0e5df;
        --taxi: #e8b93f;
        --teal: #378579;
        --coral: #ce6d55;
        --blue: #4e8ba0;
    }

    html, body, [data-testid="stAppViewContainer"] {
        color: var(--ink);
        background-color: var(--paper);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background-image: repeating-linear-gradient(0deg, transparent, transparent 31px, rgba(55, 80, 70, 0.025) 32px);
    }
    [data-testid="stHeader"] { background: rgba(245, 246, 242, 0.88); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1460px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1, h2, h3, p, label { color: var(--ink); }
    h1, h2, h3 { font-family: 'Manrope', 'DM Sans', sans-serif; }
    .dashboard-masthead {
        position: relative;
        overflow: hidden;
        margin: 0 0 1.6rem;
        padding: 1.5rem 1.8rem 1.6rem;
        border: 1px solid #dce2db;
        border-radius: 8px;
        background: linear-gradient(112deg, #fff 0%, #fff 68%, #f7f0d9 100%);
        box-shadow: 0 8px 24px rgba(37, 50, 47, 0.045);
    }
    .dashboard-masthead:after {
        position: absolute;
        top: 0;
        right: 0;
        width: 8px;
        height: 100%;
        background: var(--taxi);
        content: '';
    }
    .masthead-kicker {
        margin-bottom: 0.55rem;
        color: var(--teal);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .masthead-title {
        margin: 0;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.18;
    }
    .masthead-subtitle { margin: 0.55rem 0 0; color: var(--muted); font-size: 0.98rem; }
    .section-heading {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 1.1rem 0 0.8rem;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.16rem;
        font-weight: 800;
    }
    .section-index {
        display: inline-grid;
        width: 2rem;
        height: 2rem;
        place-items: center;
        border-radius: 6px;
        background: #e6efe9;
        color: var(--teal);
        font-size: 0.76rem;
        font-weight: 800;
    }
    [data-testid="stMetric"] {
        min-height: 116px;
        padding: 1rem 1.15rem;
        border: 1px solid var(--line);
        border-top: 3px solid var(--taxi);
        border-radius: 8px;
        background: #fff;
        box-shadow: 0 5px 18px rgba(37, 50, 47, 0.04);
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.83rem; font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Manrope', sans-serif; font-weight: 800; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 6px 22px rgba(37, 50, 47, 0.035);
    }
    [data-testid="stCaptionContainer"] p { color: var(--muted); }
    [data-testid="stPlotlyChart"] .gtitle,
    [data-testid="stPlotlyChart"] .xtick text,
    [data-testid="stPlotlyChart"] .ytick text,
    [data-testid="stPlotlyChart"] .legend text,
    [data-testid="stPlotlyChart"] .annotation-text,
    [data-testid="stPlotlyChart"] .legendtitle text {
        fill: #25322f !important;
        opacity: 1 !important;
    }
    hr { border-color: var(--line); }

    @media (max-width: 700px) {
        .block-container { padding: 1.1rem 1rem 2rem; }
        .dashboard-masthead { padding: 1.2rem 1.15rem; }
        .masthead-title { font-size: 1.55rem; }
        .masthead-subtitle { font-size: 0.9rem; }
        [data-testid="stMetric"] { min-height: 96px; padding: 0.8rem; }
    }
    </style>
    <header class="dashboard-masthead">
        <div class="masthead-kicker">NYC · Yellow Taxi · Fleet intelligence</div>
        <h1 class="masthead-title">Otimização de Frota</h1>
        <p class="masthead-subtitle">Rentabilidade e padrões de procura por zona e horário.</p>
    </header>
    """,
    unsafe_allow_html=True,
)

# 2. Carregamento dos Data Marts (com Cache)
@st.cache_data
def _carregar_gold_cache(nome_arquivo, modified_at):
    return pd.read_parquet(GOLD_DIR / nome_arquivo)


def carregar_gold(nome_arquivo):
    caminho = GOLD_DIR / nome_arquivo
    return _carregar_gold_cache(nome_arquivo, caminho.stat().st_mtime_ns)


@st.cache_data
def carregar_mapa():
    return gpd.read_parquet(ZONE_PARQUET)

df_kpis = carregar_gold("gold_kpis_gerais.parquet").iloc[0]
df_mapa = carregar_gold("gold_mapa_calor.parquet")
df_h1 = carregar_gold("gold_hipotese1_origem.parquet")
df_h2 = carregar_gold("gold_hipotese2_pagamento.parquet")
df_horarios = carregar_gold("gold_pergunta1_horarios.parquet")
df_distancia = carregar_gold("gold_pergunta2_distancia.parquet")
df_dias = carregar_gold("gold_insight_dias.parquet")
df_eficiencia = carregar_gold("gold_insight_eficiencia.parquet")
df_regional = carregar_gold("gold_insight_regional.parquet")
mapa_zonas = carregar_mapa()

CORES = {
    "ink": "#25322f",
    "muted": "#687571",
    "grid": "#e5e9e4",
    "teal": "#378579",
    "yellow": "#e8b93f",
    "coral": "#ce6d55",
    "blue": "#4e8ba0",
}
PALETA_CATEGORICA = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#332288",
]


def estilizar_figura(fig, altura=360):
    fig.update_layout(
        template="plotly_white",
        height=altura,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": CORES["ink"], "size": 13},
        colorway=PALETA_CATEGORICA,
        title={"x": 0.02, "xanchor": "left", "font": {"family": "Manrope, sans-serif", "size": 18, "color": CORES["ink"]}},
        margin={"l": 18, "r": 18, "t": 58, "b": 24},
        legend={"title": None, "font": {"size": 12, "color": CORES["ink"]}},
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#b8c2bc",
        tickfont={"color": CORES["ink"], "size": 12},
        title_font={"color": CORES["ink"], "size": 13},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#d9dfda",
        zeroline=False,
        tickfont={"color": CORES["ink"], "size": 12},
        title_font={"color": CORES["ink"], "size": 13},
    )
    fig.update_traces(textfont={"color": CORES["ink"], "size": 13})
    return fig

# 3. Painel de KPIs Superiores
col1, col2, col3 = st.columns(3)
col1.metric("Volume Total de Viagens", f"{df_kpis['volume_total']:,.0f}")
col2.metric("Receita Bruta Média", f"US$ {df_kpis['media_dolares_por_hora']:.2f} /h")
col3.metric("Gorjeta Média (Geral)", f"{df_kpis['media_gorjeta_pct']:.1f}%")

col4, col5, col6 = st.columns(3)
col4.metric("Duração Média", f"{df_kpis['duracao_media_min']:.0f} min")
col5.metric("Média US$/Milha", f"US$ {df_kpis['media_dolares_por_milha']:.2f}")
col6.metric("Velocidade Média", f"{df_kpis['velocidade_media_mph']:.1f} mph")

# 4. Mapa de Calor (Visual Principal)
st.markdown('<div class="section-heading"><span class="section-index">01</span> Receita bruta por zona</div>', unsafe_allow_html=True)
# Criar o mapa interativo
fig_mapa = px.choropleth_map(
    df_mapa,
    geojson=mapa_zonas.set_index('LocationID').geometry.__geo_interface__,
    locations='LocationID',
    color='media_dolares_por_hora',
    hover_name='Zone',
    hover_data={
        'Borough': True,
        'LocationID': True,
        'media_dolares_por_hora': ':.2f',
    },
    color_continuous_scale=[
        [0.0, "#2166AC"],
        [0.25, "#67A9CF"],
        [0.5, "#F2C14E"],
        [0.75, "#EF8A62"],
        [1.0, "#B2182B"],
    ],
    map_style="carto-positron",
    zoom=9, 
    center={"lat": 40.730610, "lon": -73.935242},
    opacity=0.7,
    labels={
        'Zone': 'Zona',
        'Borough': 'Distrito',
        'LocationID': 'ID da Zona',
        'media_dolares_por_hora': 'US$ recebido / hora',
    },
    range_color=(0, None),
)
fig_mapa.update_layout(
    height=520,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor="rgba(0,0,0,0)",
    font={"family": "DM Sans, sans-serif", "color": CORES["ink"]},
    coloraxis_colorbar={
        "title": {"text": "US$ recebido / hora", "font": {"color": "#000000", "size": 12}},
        "tickfont": {"color": "#000000", "size": 11},
        "outlinecolor": "#202020",
        "outlinewidth": 1,
    },
)
with st.container(border=True):
    st.plotly_chart(fig_mapa, width="stretch")
st.info(
    "💡 Insight Estratégico: Zonas com cores mais quentes (amarelo/verde) representam "
    "maior receita bruta por hora, antes dos custos operacionais. Desloque motoristas "
    "ociosos para as bordas destas zonas e valide a procura local."
)

# 5. Gráficos das Hipóteses e Perguntas (Lado a Lado)
st.markdown('<div class="section-heading"><span class="section-index">02</span> Padrões de procura e gorjetas</div>', unsafe_allow_html=True)

col_esq, col_dir = st.columns(2)

with col_esq:
    fig1 = px.bar(
        df_h1,
        x="Origem_Agrupada",
        y="dolares_por_hora",
        title="Receita bruta média por hora por grupo de origem",
        text_auto=".2f",
        color="Origem_Agrupada",
        color_discrete_map={"Aeroporto": CORES["yellow"], "Manhattan": CORES["teal"], "Outros": CORES["coral"]},
        labels={"Origem_Agrupada": "Origem", "dolares_por_hora": "US$ recebido / hora"},
    )
    estilizar_figura(fig1)
    with st.container(border=True):
        st.plotly_chart(fig1, width="stretch")
    grupo_top = df_h1.loc[df_h1["dolares_por_hora"].idxmax()]
    st.success(
        f"Leitura de gestão: {grupo_top['Origem_Agrupada']} lidera a receita bruta média "
        f"observada, com USD {grupo_top['dolares_por_hora']:.2f}/h. O indicador não "
        "desconta combustível, manutenção ou outros custos; cruze-o com procura e oferta."
    )

    fig3 = px.line(
        df_horarios,
        x="pickup_hour",
        y="total_viagens",
        color="Borough",
        title="Corridas por hora fora de Manhattan",
        color_discrete_sequence=PALETA_CATEGORICA,
        labels={"pickup_hour": "Hora de embarque", "total_viagens": "Viagens", "Borough": "Distrito"},
    )
    estilizar_figura(fig3, altura=380)
    fig3.update_traces(line={"width": 2.5})
    with st.container(border=True):
        st.plotly_chart(fig3, width="stretch")
    st.info(
        "Leitura de gestão: os picos horários ajudam a programar turnos e reposicionar "
        "veículos antes dos períodos de maior procura fora de Manhattan."
    )

with col_dir:
    fig2 = px.bar(
        df_h2,
        x="metodo_pagamento",
        y="perc_gorjeta",
        title="Gorjeta média por método de pagamento",
        text_auto=".2f",
        color="metodo_pagamento",
        color_discrete_map={"Cartão de Crédito": CORES["teal"], "Dinheiro": CORES["coral"]},
        labels={"metodo_pagamento": "Pagamento", "perc_gorjeta": "Gorjeta média (%)"},
    )
    estilizar_figura(fig2)
    with st.container(border=True):
        st.plotly_chart(fig2, width="stretch")
    st.info(
        "Leitura de gestão: compare os percentuais como comportamento de gorjeta "
        "registrado por método; não interprete esta métrica isoladamente como receita total."
    )
    st.warning(
        "Viés técnico: o sistema TLC não regista automaticamente as gorjetas pagas em "
        "dinheiro. Por isso, a média de gorjeta para dinheiro aparece subestimada e não é "
        "diretamente comparável à média do cartão."
    )

    fig4 = px.bar(
        df_distancia,
        x="faixa_distancia",
        y="percentual_corridas_gorjeta_acima_20",
        title="Corridas com gorjeta acima de 20%",
        text_auto=".1f",
        color="faixa_distancia",
        color_discrete_sequence=PALETA_CATEGORICA,
        labels={"faixa_distancia": "Distância", "percentual_corridas_gorjeta_acima_20": "Corridas (%)"},
    )
    estilizar_figura(fig4)
    with st.container(border=True):
        st.plotly_chart(fig4, width="stretch")
    st.info(
        "Leitura de gestão: a taxa por faixa revela em quais distâncias gorjetas acima "
        "de 20% são mais frequentes; use-a como sinal de perfil da viagem, não como garantia "
        "de retorno para uma corrida específica."
    )

st.markdown(
    '<div class="section-heading"><span class="section-index">03</span> '
    'Eficiência de Frota e Sazonalidade</div>',
    unsafe_allow_html=True,
)
col_dias, col_eficiencia = st.columns(2)

with col_dias:
    fig_dias = px.bar(
        df_dias,
        x="pickup_day_name",
        y="media_dolares_por_hora",
        title="Receita bruta média por dia da semana",
        text_auto=".2f",
        color="pickup_day_name",
        category_orders={
            "pickup_day_name": [
                "Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira",
                "Sexta-feira", "Sábado", "Domingo",
            ]
        },
        color_discrete_sequence=PALETA_CATEGORICA[:7],
        labels={
            "pickup_day_name": "Dia da semana",
            "media_dolares_por_hora": "US$ recebido / hora",
        },
    )
    estilizar_figura(fig_dias)
    fig_dias.update_layout(showlegend=False)
    with st.container(border=True):
        st.plotly_chart(fig_dias, width="stretch")
    melhor_dia = df_dias.loc[df_dias["media_dolares_por_hora"].idxmax()]
    st.info(
        f"Leitura de gestão: {melhor_dia['pickup_day_name']} apresenta a maior receita "
        f"bruta média, "
        f"USD {melhor_dia['media_dolares_por_hora']:.2f}/h. Compare o padrão semanal ao "
        "planejar folgas e reforços de frota; o valor não representa lucro líquido."
    )

with col_eficiencia:
    fig_eficiencia = px.scatter(
        df_regional,
        x="velocidade_media_mph",
        y="media_dolares_por_milha",
        color="Area_Operacional",
        title="Velocidade e receita por milha em cada área",
        color_discrete_map={
            "Aeroporto": CORES["yellow"],
            "Manhattan": CORES["teal"],
            "Queens": CORES["blue"],
            "Brooklyn": CORES["coral"],
            "Bronx": "#788c68",
            "Staten Island": "#9a725d",
        },
        labels={
            "Area_Operacional": "Área",
            "velocidade_media_mph": "Velocidade média (mph)",
            "media_dolares_por_milha": "US$ médio por milha",
        },
        hover_data={
            "quantidade_corridas": ":,",
            "velocidade_media_mph": ":.1f",
            "media_dolares_por_milha": ":.2f",
        },
    )
    estilizar_figura(fig_eficiencia)
    fig_eficiencia.update_traces(
        marker={"size": 18, "line": {"width": 1.5, "color": "white"}},
        textposition="top center",
    )
    fig_eficiencia.update_xaxes(title_text="Velocidade média (mph)")
    fig_eficiencia.update_yaxes(title_text="US$ médio por milha")
    with st.container(border=True):
        st.plotly_chart(fig_eficiencia, width="stretch")

    aeroporto = df_eficiencia.set_index("Origem_Agrupada").loc["Aeroporto"]
    manhattan = df_eficiencia.set_index("Origem_Agrupada").loc["Manhattan"]
    receita_hora = df_h1.set_index("Origem_Agrupada")["dolares_por_hora"]
    st.info(
        f"Como ler: aeroportos têm velocidade média de {aeroporto['velocidade_media_mph']:.1f} mph "
        f"e USD {aeroporto['media_dolares_por_milha']:.2f}/milha; Manhattan tem "
        f"{manhattan['velocidade_media_mph']:.1f} mph e USD {manhattan['media_dolares_por_milha']:.2f}/milha. "
        f"A maior velocidade acompanha maior receita bruta/h nos aeroportos "
        f"(USD {receita_hora['Aeroporto']:.2f} contra USD {receita_hora['Manhattan']:.2f}), "
        "apesar da menor receita por milha. Por viagem, receita/h = receita/milha × mph; "
        "o produto das médias dos eixos não é igual à média de receita/h. Receita bruta "
        "não desconta custos."
    )

regional_por_receita = df_regional.sort_values("media_dolares_por_hora", ascending=True)
fig_receita_regional = px.bar(
    regional_por_receita,
    x="media_dolares_por_hora",
    y="Area_Operacional",
    orientation="h",
    color="Area_Operacional",
    text_auto=".2f",
    title="Receita bruta média por hora: comparação entre áreas",
    color_discrete_map={
        "Aeroporto": CORES["yellow"],
        "Manhattan": CORES["teal"],
        "Queens": CORES["blue"],
        "Brooklyn": CORES["coral"],
        "Bronx": "#788c68",
        "Staten Island": "#9a725d",
    },
    labels={
        "Area_Operacional": "Área",
        "media_dolares_por_hora": "US$ recebido / hora",
        "quantidade_corridas": "Corridas válidas",
    },
    hover_data={"quantidade_corridas": ":,"},
)
estilizar_figura(fig_receita_regional, altura=420)
fig_receita_regional.update_layout(showlegend=False)
fig_receita_regional.update_xaxes(title_text="US$ recebido / hora")
fig_receita_regional.update_yaxes(title_text="", categoryorder="total ascending")
with st.container(border=True):
    st.plotly_chart(fig_receita_regional, width="stretch")

st.info(
    "Leitura regional: cada barra mostra a média por corrida; passe o cursor para ver o "
    "volume que sustenta cada estimativa. Staten Island tem uma amostra pequena neste "
    "recorte, então a média é mais instável. A comparação é de receita bruta, não de lucro "
    "nem de retorno ajustado por custos."
)