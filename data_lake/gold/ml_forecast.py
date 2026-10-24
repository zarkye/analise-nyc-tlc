from pathlib import Path

import pandas as pd
from prophet import Prophet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data_lake" / "gold"
TIMESERIES_PATH = GOLD_DIR / "gold_timeseries_diario.parquet"
FORECAST_PATH = GOLD_DIR / "gold_forecast_2027.parquet"
FORECAST_END = pd.Timestamp("2027-12-31")


def build_forecast() -> pd.DataFrame:
    history = pd.read_parquet(TIMESERIES_PATH)[["ds", "y"]].copy()
    history["ds"] = pd.to_datetime(history["ds"])
    history = history.dropna(subset=["ds", "y"]).sort_values("ds")
    if len(history) < 2:
        raise ValueError(
            f"A série temporal precisa de ao menos duas observações válidas: {TIMESERIES_PATH}"
        )

    last_observation = history["ds"].max().normalize()
    if last_observation > FORECAST_END:
        raise ValueError(f"A última observação ({last_observation.date()}) passa do horizonte definido")

    model = Prophet()
    model.fit(history)

    future = model.make_future_dataframe(
        periods=(FORECAST_END - last_observation).days,
        freq="D",
    )
    forecast = model.predict(future)
    forecast["yhat_otimizado"] = forecast["yhat"] * 1.15
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "yhat_otimizado"]]

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(FORECAST_PATH, index=False)
    return result


if __name__ == "__main__":
    predictions = build_forecast()
    print(f"Previsão gerada: {len(predictions):,} dias em {FORECAST_PATH}")