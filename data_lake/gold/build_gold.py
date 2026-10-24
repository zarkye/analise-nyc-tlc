import calendar
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR = PROJECT_ROOT / "data_lake" / "bronze"
GOLD_DIR = PROJECT_ROOT / "data_lake" / "gold"
BATCH_SIZE = 250_000
SOURCE_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "payment_type",
    "PULocationID",
    "DOLocationID",
    "tip_amount",
]
MEAN_METRICS = {
    "dolares_por_hora": "media_dolares_por_hora",
    "perc_gorjeta": "media_gorjeta_pct",
    "trip_duration_min": "duracao_media_min",
    "dolares_por_milha": "media_dolares_por_milha",
    "velocidade_mph": "velocidade_media_mph",
}
DAY_NAMES = {
    "Monday": "Segunda-feira",
    "Tuesday": "Terça-feira",
    "Wednesday": "Quarta-feira",
    "Thursday": "Quinta-feira",
    "Friday": "Sexta-feira",
    "Saturday": "Sábado",
    "Sunday": "Domingo",
}
BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"]
REGIONAL_ORDER = ["Aeroporto", *BOROUGHS]
DISTANCE_LABELS = [
    "0 a 2 milhas",
    "2 a 5 milhas",
    "5 a 10 milhas",
    "Mais de 10 milhas",
]


def _clean_batch(
    df: pd.DataFrame,
    zones: pd.DataFrame,
    history_start: pd.Timestamp,
    history_end: pd.Timestamp,
) -> pd.DataFrame:
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df = df[
        (df["tpep_pickup_datetime"] >= history_start)
        & (df["tpep_pickup_datetime"] < history_end)
    ].copy()

    df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)].copy()
    df = df[df["payment_type"].isin([1, 2])]
    df = df.dropna(subset=["PULocationID", "DOLocationID", "trip_distance"])
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[df["trip_duration_min"].between(1, 180, inclusive="neither")].copy()

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["dinheiro_recebido"] = df["fare_amount"] + df["tip_amount"]
    df["dolares_por_hora"] = df["dinheiro_recebido"] / df["trip_duration_min"] * 60
    df = df[df["dolares_por_hora"].between(10, 200, inclusive="both")].copy()
    df["perc_gorjeta"] = df["tip_amount"] / df["fare_amount"] * 100
    return df.merge(
        zones[["LocationID", "Borough", "Zone"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    )


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["dolares_por_milha"] = df["dinheiro_recebido"] / df["trip_distance"]
    df["velocidade_mph"] = df["trip_distance"] / (df["trip_duration_min"] / 60)
    df[["dolares_por_milha", "velocidade_mph"]] = df[
        ["dolares_por_milha", "velocidade_mph"]
    ].replace([np.inf, -np.inf], np.nan)

    airport = df["Zone"].str.contains("Airport", case=False, na=False)
    df["Origem_Agrupada"] = np.select(
        [airport, df["Borough"].eq("Manhattan")],
        ["Aeroporto", "Manhattan"],
        default="Outros",
    )
    df["Area_Operacional"] = np.select(
        [airport, df["Borough"].isin(BOROUGHS)],
        ["Aeroporto", df["Borough"]],
        default="Fora do escopo",
    )
    df["pickup_day_order"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_day_name"] = df["tpep_pickup_datetime"].dt.day_name().map(DAY_NAMES)
    df["ds"] = df["tpep_pickup_datetime"].dt.normalize()
    df["faixa_distancia"] = pd.cut(
        df["trip_distance"],
        bins=[0, 2, 5, 10, np.inf],
        labels=DISTANCE_LABELS,
        include_lowest=True,
    )
    df["gorjeta_acima_20"] = df["perc_gorjeta"] > 20
    return df


def _mean_partials(
    df: pd.DataFrame,
    dimensions: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    aggregations = {}
    for metric in metrics:
        aggregations[f"{metric}_sum"] = (metric, "sum")
        aggregations[f"{metric}_count"] = (metric, "count")
    return df.groupby(dimensions, as_index=False, observed=True).agg(**aggregations)


def _reduce_means(
    partials: list[pd.DataFrame],
    dimensions: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    columns = [
        column
        for metric in metrics
        for column in (f"{metric}_sum", f"{metric}_count")
    ]
    combined = pd.concat(partials, ignore_index=True)
    reduced = combined.groupby(dimensions, as_index=False, observed=True)[columns].sum()
    for metric in metrics:
        reduced[metric] = reduced[f"{metric}_sum"] / reduced[f"{metric}_count"]
    return reduced


def _collect_partials(df: pd.DataFrame, partials: dict[str, list[pd.DataFrame]]) -> None:
    partials["timeseries"].append(_mean_partials(df.assign(ds=df["ds"]), ["ds"], ["dolares_por_hora"]))

    map_partial = _mean_partials(df, ["LocationID"], ["dolares_por_hora"])
    map_labels = df.groupby("LocationID", as_index=False, observed=True).agg(
        quantidade_viagens=("dolares_por_hora", "size"),
        Zone=("Zone", "first"),
        Borough=("Borough", "first"),
    )
    partials["mapa"].append(map_partial.merge(map_labels, on="LocationID", how="left"))

    partials["dias"].append(
        _mean_partials(df, ["pickup_day_order", "pickup_day_name"], ["dolares_por_hora"])
    )
    efficiency = df[df["Origem_Agrupada"].isin(["Aeroporto", "Manhattan"])]
    if not efficiency.empty:
        partials["eficiencia"].append(
            _mean_partials(
                efficiency,
                ["Origem_Agrupada"],
                ["dolares_por_milha", "velocidade_mph"],
            )
        )

    regional = df[df["Area_Operacional"] != "Fora do escopo"]
    if not regional.empty:
        regional_partial = _mean_partials(
            regional,
            ["Area_Operacional"],
            ["dolares_por_hora", "dolares_por_milha", "velocidade_mph"],
        )
        regional_count = regional.groupby(
            "Area_Operacional", as_index=False, observed=True
        ).size().rename(columns={"size": "quantidade_corridas"})
        partials["regional"].append(
            regional_partial.merge(regional_count, on="Area_Operacional", how="left")
        )

    partials["hipotese1"].append(
        _mean_partials(df, ["Origem_Agrupada"], ["dolares_por_hora"])
    )
    partials["hipotese2"].append(
        _mean_partials(df, ["payment_type"], ["perc_gorjeta"])
    )

    outside_manhattan = df[df["Borough"].isin(["Brooklyn", "Queens", "Bronx", "Staten Island"])]
    if not outside_manhattan.empty:
        partials["horarios"].append(
            outside_manhattan.groupby(
                ["Borough", "pickup_hour"], as_index=False, observed=True
            ).size().rename(columns={"size": "total_viagens"})
        )

    distance_partial = df.groupby(
        "faixa_distancia", as_index=False, observed=True
    ).agg(
        quantidade_corridas=("gorjeta_acima_20", "size"),
        gorjetas_acima_20=("gorjeta_acima_20", "sum"),
    )
    partials["distancia"].append(distance_partial)


def _build_marts(partials: dict[str, list[pd.DataFrame]], global_partials: list[dict]) -> dict[str, pd.DataFrame]:
    global_totals = pd.DataFrame(global_partials).sum(numeric_only=True)
    kpis = {
        "volume_total": int(global_totals["volume_total"]),
    }
    for metric, output_name in MEAN_METRICS.items():
        kpis[output_name] = global_totals[f"{metric}_sum"] / global_totals[f"{metric}_count"]

    timeseries = _reduce_means(partials["timeseries"], ["ds"], ["dolares_por_hora"])
    map_data = pd.concat(partials["mapa"], ignore_index=True)
    map_data = map_data.groupby("LocationID", as_index=False, observed=True).agg(
        quantidade_viagens=("quantidade_viagens", "sum"),
        dolares_por_hora_sum=("dolares_por_hora_sum", "sum"),
        dolares_por_hora_count=("dolares_por_hora_count", "sum"),
        Zone=("Zone", "first"),
        Borough=("Borough", "first"),
    )
    map_data["media_dolares_por_hora"] = (
        map_data["dolares_por_hora_sum"] / map_data["dolares_por_hora_count"]
    )
    map_data = map_data[
        ["LocationID", "quantidade_viagens", "media_dolares_por_hora", "Zone", "Borough"]
    ]

    day_data = _reduce_means(
        partials["dias"], ["pickup_day_order", "pickup_day_name"], ["dolares_por_hora"]
    )
    day_data = day_data.rename(columns={"dolares_por_hora": "media_dolares_por_hora"})
    day_data = day_data.sort_values("pickup_day_order").drop(columns="pickup_day_order")

    efficiency = _reduce_means(
        partials["eficiencia"],
        ["Origem_Agrupada"],
        ["dolares_por_milha", "velocidade_mph"],
    ).rename(
        columns={
            "dolares_por_milha": "media_dolares_por_milha",
            "velocidade_mph": "velocidade_media_mph",
        }
    )

    regional_frames = pd.concat(partials["regional"], ignore_index=True)
    regional = regional_frames.groupby("Area_Operacional", as_index=False, observed=True).agg(
        quantidade_corridas=("quantidade_corridas", "sum"),
        dolares_por_hora_sum=("dolares_por_hora_sum", "sum"),
        dolares_por_hora_count=("dolares_por_hora_count", "sum"),
        dolares_por_milha_sum=("dolares_por_milha_sum", "sum"),
        dolares_por_milha_count=("dolares_por_milha_count", "sum"),
        velocidade_mph_sum=("velocidade_mph_sum", "sum"),
        velocidade_mph_count=("velocidade_mph_count", "sum"),
    )
    for metric in ["dolares_por_hora", "dolares_por_milha", "velocidade_mph"]:
        regional[metric] = regional[f"{metric}_sum"] / regional[f"{metric}_count"]
    regional = regional.rename(
        columns={
            "dolares_por_hora": "media_dolares_por_hora",
            "dolares_por_milha": "media_dolares_por_milha",
            "velocidade_mph": "velocidade_media_mph",
        }
    )
    regional["ordem"] = regional["Area_Operacional"].map(
        {area: index for index, area in enumerate(REGIONAL_ORDER)}
    )
    regional = regional.sort_values("ordem").drop(columns="ordem")
    regional = regional[
        [
            "Area_Operacional",
            "quantidade_corridas",
            "media_dolares_por_hora",
            "media_dolares_por_milha",
            "velocidade_media_mph",
        ]
    ]

    hypothesis_origin = _reduce_means(
        partials["hipotese1"], ["Origem_Agrupada"], ["dolares_por_hora"]
    )[["Origem_Agrupada", "dolares_por_hora"]]
    hypothesis_payment = _reduce_means(
        partials["hipotese2"], ["payment_type"], ["perc_gorjeta"]
    )[["payment_type", "perc_gorjeta"]]
    hypothesis_payment["metodo_pagamento"] = hypothesis_payment["payment_type"].map(
        {1: "Cartão de Crédito", 2: "Dinheiro"}
    )

    hours = pd.concat(partials["horarios"], ignore_index=True)
    hours = hours.groupby(["Borough", "pickup_hour"], as_index=False, observed=True)[
        "total_viagens"
    ].sum()

    distance = pd.concat(partials["distancia"], ignore_index=True)
    distance = distance.groupby("faixa_distancia", observed=True).sum()
    distance = distance.reindex(DISTANCE_LABELS)
    distance["percentual_corridas_gorjeta_acima_20"] = (
        distance["gorjetas_acima_20"] / distance["quantidade_corridas"] * 100
    )
    distance = distance["percentual_corridas_gorjeta_acima_20"].rename_axis(
        "faixa_distancia"
    ).reset_index()

    return {
        "gold_kpis_gerais.parquet": pd.DataFrame([kpis]),
        "gold_mapa_calor.parquet": map_data,
        "gold_timeseries_diario.parquet": timeseries[["ds", "dolares_por_hora"]].rename(
            columns={"dolares_por_hora": "y"}
        ),
        "gold_insight_dias.parquet": day_data,
        "gold_insight_eficiencia.parquet": efficiency,
        "gold_insight_regional.parquet": regional,
        "gold_hipotese1_origem.parquet": hypothesis_origin,
        "gold_hipotese2_pagamento.parquet": hypothesis_payment,
        "gold_pergunta1_horarios.parquet": hours,
        "gold_pergunta2_distancia.parquet": distance,
    }


def build_gold() -> dict[str, pd.DataFrame]:
    parquet_files = sorted(BRONZE_DIR.glob("dados_taxi_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Nenhum arquivo dados_taxi_*.parquet encontrado em {BRONZE_DIR}")
    available_periods = []
    month_numbers = {
        name.lower(): month
        for month in range(1, 13)
        for name in (calendar.month_name[month], calendar.month_abbr[month])
    }
    for path in parquet_files:
        _, _, year, month_name = path.stem.split("_", maxsplit=3)
        month = month_numbers[month_name.lower()]
        available_periods.append((int(year), month))
    first_year, first_month = min(available_periods)
    last_year, last_month = max(available_periods)
    history_start = pd.Timestamp(datetime(first_year, first_month, 1))
    if last_month == 12:
        history_end = pd.Timestamp(datetime(last_year + 1, 1, 1))
    else:
        history_end = pd.Timestamp(datetime(last_year, last_month + 1, 1))
    zones = pd.read_csv(BRONZE_DIR / "taxi_zone_lookup.csv")
    partials: dict[str, list[pd.DataFrame]] = {
        name: []
        for name in [
            "timeseries", "mapa", "dias", "eficiencia", "regional", "hipotese1",
            "hipotese2", "horarios", "distancia",
        ]
    }
    global_partials = []

    for path in parquet_files:
        parquet_file = pq.ParquetFile(path)
        missing_columns = set(SOURCE_COLUMNS) - set(parquet_file.schema_arrow.names)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Colunas ausentes em {path.name}: {missing}")

        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=SOURCE_COLUMNS):
            cleaned = _clean_batch(batch.to_pandas(), zones, history_start, history_end)
            if cleaned.empty:
                continue
            enriched = _add_derived_columns(cleaned)
            _collect_partials(enriched, partials)

            global_row = {"volume_total": len(enriched)}
            for metric in MEAN_METRICS:
                global_row[f"{metric}_sum"] = enriched[metric].sum()
                global_row[f"{metric}_count"] = enriched[metric].count()
            global_partials.append(global_row)

    if not global_partials:
        raise ValueError("Nenhuma viagem válida foi encontrada nas partições Bronze")

    marts = _build_marts(partials, global_partials)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    for filename, mart in marts.items():
        mart.to_parquet(GOLD_DIR / filename, index=False)
    return marts


if __name__ == "__main__":
    generated = build_gold()
    for filename, mart in generated.items():
        print(f"Gerado {GOLD_DIR / filename}: {len(mart):,} linhas")