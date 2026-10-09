from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR = PROJECT_ROOT / "data_lake" / "bronze"
SILVER_DIR = PROJECT_ROOT / "data_lake" / "silver"
ZONE_SHAPEFILE = PROJECT_ROOT / "geo_data" / "taxi_zones" / "taxi_zones.shp"
ZONE_PARQUET = PROJECT_ROOT / "geo_data" / "taxi_zones" / "taxi_zones.parquet"


def build_silver() -> pd.DataFrame:
    df = pd.read_parquet(BRONZE_DIR / "dados_taxi.parquet")
    df_zones = pd.read_csv(BRONZE_DIR / "taxi_zone_lookup.csv")

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

    df = df.merge(
        df_zones[["LocationID", "Borough", "Zone"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    )

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SILVER_DIR / "taxi_limpo.parquet", index=False)
    return df


def build_zone_geometry() -> None:
    zones = gpd.read_file(ZONE_SHAPEFILE).to_crs(epsg=4326)
    ZONE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    zones.to_parquet(ZONE_PARQUET, index=False)


if __name__ == "__main__":
    silver = build_silver()
    build_zone_geometry()
    print(f"Silver gerada: {len(silver):,} viagens em {SILVER_DIR / 'taxi_limpo.parquet'}")
    print(f"Geometria das zonas: {ZONE_PARQUET}")