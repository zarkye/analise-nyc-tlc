from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_PATH = PROJECT_ROOT / "data_lake" / "silver" / "taxi_limpo.parquet"
GOLD_DIR = PROJECT_ROOT / "data_lake" / "gold"


def build_gold() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(SILVER_PATH)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    marts = {
        "gold_kpis_gerais.parquet": pd.DataFrame(
            [
                {
                    "volume_total": len(df),
                    "media_dolares_por_hora": df["dolares_por_hora"].mean(),
                    "media_gorjeta_pct": df["perc_gorjeta"].mean(),
                    "duracao_media_min": df["trip_duration_min"].mean(),
                }
            ]
        ),
        "gold_mapa_calor.parquet": df.groupby("LocationID", as_index=False).agg(
            media_dolares_por_hora=("dolares_por_hora", "mean"),
            Zone=("Zone", "first"),
            Borough=("Borough", "first"),
        ),
    }

    airport = df["Zone"].str.contains("Airport", case=False, na=False)
    df["Origem_Agrupada"] = np.select(
        [airport, df["Borough"].eq("Manhattan")],
        ["Aeroporto", "Manhattan"],
        default="Outros",
    )
    marts["gold_hipotese1_origem.parquet"] = (
        df.groupby("Origem_Agrupada", as_index=False)["dolares_por_hora"].mean()
    )

    marts["gold_hipotese2_pagamento.parquet"] = (
        df.groupby("payment_type", as_index=False)["perc_gorjeta"]
        .mean()
        .assign(
            metodo_pagamento=lambda result: result["payment_type"].map(
                {1: "Cartão de Crédito", 2: "Dinheiro"}
            )
        )
    )

    boroughs_outside_manhattan = ["Brooklyn", "Queens", "Bronx", "Staten Island"]
    marts["gold_pergunta1_horarios.parquet"] = (
        df[df["Borough"].isin(boroughs_outside_manhattan)]
        .groupby(["Borough", "pickup_hour"])
        .size()
        .reset_index(name="total_viagens")
    )

    distance_bins = [0, 2, 5, 10, np.inf]
    distance_labels = ["0 a 2 milhas", "2 a 5 milhas", "5 a 10 milhas", "Mais de 10 milhas"]
    df["faixa_distancia"] = pd.cut(
        df["trip_distance"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
    )
    df["gorjeta_acima_20"] = df["perc_gorjeta"] > 20
    marts["gold_pergunta2_distancia.parquet"] = (
        df.groupby("faixa_distancia", observed=False)["gorjeta_acima_20"]
        .mean()
        .mul(100)
        .rename("percentual_corridas_gorjeta_acima_20")
        .reset_index()
    )

    for filename, mart in marts.items():
        mart.to_parquet(GOLD_DIR / filename, index=False)

    return marts


if __name__ == "__main__":
    generated = build_gold()
    for filename, mart in generated.items():
        print(f"Gerado {GOLD_DIR / filename}: {len(mart):,} linhas")