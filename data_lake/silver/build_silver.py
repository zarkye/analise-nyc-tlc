from pathlib import Path

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR = PROJECT_ROOT / "data_lake" / "bronze"
SILVER_DIR = PROJECT_ROOT / "data_lake" / "silver"
ZONE_SHAPEFILE = PROJECT_ROOT / "geo_data" / "taxi_zones" / "taxi_zones.shp"
ZONE_PARQUET = PROJECT_ROOT / "geo_data" / "taxi_zones" / "taxi_zones.parquet"
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


def _clean_batch(df: pd.DataFrame, df_zones: pd.DataFrame) -> pd.DataFrame:
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

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
        df_zones[["LocationID", "Borough", "Zone"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    )


def build_silver() -> int:
    parquet_files = sorted(BRONZE_DIR.glob("dados_taxi_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Nenhum arquivo dados_taxi_*.parquet encontrado em {BRONZE_DIR}")
    df_zones = pd.read_csv(BRONZE_DIR / "taxi_zone_lookup.csv")
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SILVER_DIR / "taxi_limpo.parquet"
    temporary_path = SILVER_DIR / "taxi_limpo.parquet.tmp"
    writer = None
    total_rows = 0

    try:
        for path in parquet_files:
            parquet_file = pq.ParquetFile(path)
            missing_columns = set(SOURCE_COLUMNS) - set(parquet_file.schema_arrow.names)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"Colunas ausentes em {path.name}: {missing}")

            for batch in parquet_file.iter_batches(
                batch_size=BATCH_SIZE,
                columns=SOURCE_COLUMNS,
            ):
                cleaned = _clean_batch(batch.to_pandas(), df_zones)
                if cleaned.empty:
                    continue

                table = pa.Table.from_pandas(cleaned, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary_path, table.schema)
                elif table.schema != writer.schema:
                    table = table.cast(writer.schema, safe=False)
                writer.write_table(table)
                total_rows += len(cleaned)

        if writer is None:
            raise ValueError("Nenhuma viagem válida foi encontrada nas partições Bronze")
        writer.close()
        writer = None
        temporary_path.replace(output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    return total_rows


def build_zone_geometry() -> None:
    zones = gpd.read_file(ZONE_SHAPEFILE).to_crs(epsg=4326)
    ZONE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    zones.to_parquet(ZONE_PARQUET, index=False)


if __name__ == "__main__":
    silver_rows = build_silver()
    build_zone_geometry()
    print(f"Silver gerada: {silver_rows:,} viagens em {SILVER_DIR / 'taxi_limpo.parquet'}")
    print(f"Geometria das zonas: {ZONE_PARQUET}")