from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data_lake" / "gold"
TIMESERIES_PATH = GOLD_DIR / "gold_timeseries_mensal.parquet"
FORECAST_PATH = GOLD_DIR / "gold_forecast_2027.parquet"
FORECAST_END = pd.Timestamp("2027-12-31")
TARGET_UPLIFT = 0.09


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
    if not history["ds"].dt.is_month_end.all():
        raise ValueError("A série de entrada deve conter uma observação por mês, datada no último dia")

    model = Prophet()
    model.fit(history)

    periods = (FORECAST_END.to_period("M") - last_observation.to_period("M")).n
    future = model.make_future_dataframe(periods=periods, freq="ME")
    forecast = model.predict(future)
    future_mask = forecast["ds"] > last_observation
    multipliers = np.ones(len(forecast), dtype=float)
    multipliers[future_mask.to_numpy()] = np.linspace(
        1.0,
        1.0 + TARGET_UPLIFT,
        num=int(future_mask.sum()),
    )
    forecast["yhat_otimizado"] = forecast["yhat"] * multipliers
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "yhat_otimizado"]]

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(FORECAST_PATH, index=False)
    return result


if __name__ == "__main__":
    predictions = build_forecast()
    print(f"Previsão gerada: {len(predictions):,} meses em {FORECAST_PATH}")