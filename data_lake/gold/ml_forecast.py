from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data_lake" / "gold"
TIMESERIES_PATH = GOLD_DIR / "gold_timeseries_mensal.parquet"
FORECAST_PATH = GOLD_DIR / "gold_forecast_2027.parquet"
VOLUME_TIMESERIES_PATH = GOLD_DIR / "gold_timeseries_volume_bairro.parquet"
VOLUME_FORECAST_PATH = GOLD_DIR / "gold_forecast_volume_bairro_2027.parquet"
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


def build_volume_forecast() -> pd.DataFrame:
    history = pd.read_parquet(VOLUME_TIMESERIES_PATH)[["ds", "Borough", "y"]].copy()
    history["ds"] = pd.to_datetime(history["ds"])
    history["Borough"] = history["Borough"].astype("string").str.strip()
    history = history.dropna(subset=["ds", "Borough", "y"])
    history = history[history["Borough"].str.casefold() != "unknown"]
    history = history.sort_values(["Borough", "ds"])
    if history.empty:
        raise ValueError(f"Nenhuma série mensal válida encontrada em {VOLUME_TIMESERIES_PATH}")
    if not history["ds"].dt.is_month_start.all():
        raise ValueError("A série de volume deve conter observações datadas no primeiro dia do mês")

    forecasts = []
    for borough in sorted(history["Borough"].unique()):
        borough_history = history.loc[history["Borough"] == borough, ["ds", "y"]]
        if len(borough_history) < 2:
            raise ValueError(f"O bairro {borough} precisa de ao menos duas observações mensais")

        last_observation = borough_history["ds"].max().normalize()
        if last_observation > FORECAST_END:
            raise ValueError(
                f"A última observação de {borough} ({last_observation.date()}) passa do horizonte definido"
            )

        model = Prophet()
        model.fit(borough_history)
        periods = (FORECAST_END.to_period("M") - last_observation.to_period("M")).n
        future = model.make_future_dataframe(periods=periods, freq="MS")
        borough_forecast = model.predict(future)
        borough_forecast["Borough"] = borough
        forecasts.append(borough_forecast)

    result = pd.concat(forecasts, ignore_index=True)
    result = result[["ds", "Borough", "yhat", "yhat_lower", "yhat_upper"]]
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(VOLUME_FORECAST_PATH, index=False)
    return result


if __name__ == "__main__":
    predictions = build_forecast()
    print(f"Previsão gerada: {len(predictions):,} meses em {FORECAST_PATH}")
    volume_predictions = build_volume_forecast()
    print(
        f"Previsão de volume gerada: {len(volume_predictions):,} registros em "
        f"{VOLUME_FORECAST_PATH}"
    )