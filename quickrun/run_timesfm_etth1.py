from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import timesfm

# ============================================================
# 实验配置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

DATA_PATH = BASE_DIR / "ETTh1.csv"

CONTEXT = 512       # 使用过去 512 小时
HORIZON = 96        # 预测未来 96 小时
QUANTILES = np.arange(0.1, 1.0, 0.1)

# ============================================================
# 读取数据：只预测 OT 油温
# ============================================================
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

values = df["OT"].astype("float32").to_numpy()
timestamps = df["date"]

context = values[-(CONTEXT + HORIZON):-HORIZON]
y_true = values[-HORIZON:]
test_time = timestamps.iloc[-HORIZON:].reset_index(drop=True)

print("Context length:", len(context))
print("Forecast horizon:", len(y_true))
print("Test period:", test_time.iloc[0], "->", test_time.iloc[-1])

# ============================================================
# 评价函数
# ============================================================
def pinball_loss(y, qhat, tau):
    error = y - qhat
    return np.maximum(tau * error, (tau - 1.0) * error).mean()

def point_metrics(y, pred):
    mae = np.mean(np.abs(y - pred))
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    return mae, rmse

# ============================================================
# 简单基线：重复前一天的 24 小时模式
# ============================================================
naive_pred = np.resize(context[-24:], HORIZON)
naive_mae, naive_rmse = point_metrics(y_true, naive_pred)

# ============================================================
# TimesFM 2.5 零样本预测
# ============================================================
torch.set_float32_matmul_precision("high")

print("\nLoading TimesFM 2.5 model...")
print("The first run may download model weights from Hugging Face.")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

model.compile(
    timesfm.ForecastConfig(
        max_context=CONTEXT,
        max_horizon=HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

start = time.perf_counter()

point_forecast, quantile_forecast = model.forecast(
    horizon=HORIZON,
    inputs=[context],
)

elapsed = time.perf_counter() - start

point_pred = np.asarray(point_forecast)[0].astype("float32")

# 官方接口输出结构：
# 第 0 列为 mean，后续 9 列为 P10, P20, ..., P90
all_quantiles = np.asarray(quantile_forecast)[0]
quantile_pred = all_quantiles[:, 1:10].astype("float32")

assert point_pred.shape == (HORIZON,)
assert quantile_pred.shape == (HORIZON, 9)

timesfm_mae, timesfm_rmse = point_metrics(y_true, point_pred)

ql_values = np.array(
    [
        pinball_loss(y_true, quantile_pred[:, i], tau)
        for i, tau in enumerate(QUANTILES)
    ]
)

mean_ql = ql_values.mean()

# 基于 P10 到 P90 的 Quantile Loss 数值积分近似。
# 该指标只覆盖中央分位数区间，不应写成完整精确 CRPS。
crps_q10_q90_approx = 2.0 * np.trapezoid(ql_values, QUANTILES)

lower = quantile_pred[:, 0]
upper = quantile_pred[:, -1]

coverage = np.mean((y_true >= lower) & (y_true <= upper))
interval_width = np.mean(upper - lower)

# ============================================================
# 保存指标结果
# ============================================================
metrics = pd.DataFrame(
    [
        {
            "model": "SeasonalNaive_24h",
            "MAE": naive_mae,
            "RMSE": naive_rmse,
            "MeanQuantileLoss": np.nan,
            "CRPS_Q10_Q90_approx": np.nan,
            "P10_P90_Coverage": np.nan,
            "P10_P90_Width": np.nan,
            "InferenceSeconds": 0.0,
        },
        {
            "model": "TimesFM_2.5_200M",
            "MAE": timesfm_mae,
            "RMSE": timesfm_rmse,
            "MeanQuantileLoss": mean_ql,
            "CRPS_Q10_Q90_approx": crps_q10_q90_approx,
            "P10_P90_Coverage": coverage,
            "P10_P90_Width": interval_width,
            "InferenceSeconds": elapsed,
        },
    ]
)

metrics.to_csv(OUT_DIR / "metrics.csv", index=False)

# ============================================================
# 保存逐时预测结果
# ============================================================
predictions = pd.DataFrame(
    {
        "timestamp": test_time,
        "actual": y_true,
        "point_prediction": point_pred,
        "seasonal_naive_prediction": naive_pred,
    }
)

for i, tau in enumerate(QUANTILES):
    predictions[f"q{int(round(tau * 100)):02d}"] = quantile_pred[:, i]

predictions.to_csv(OUT_DIR / "forecast_values.csv", index=False)

# ============================================================
# 绘图
# ============================================================
history_time = timestamps.iloc[-(HORIZON + 192):-HORIZON]
history_values = values[-(HORIZON + 192):-HORIZON]

plt.figure(figsize=(14, 5))
plt.plot(history_time, history_values, label="Historical OT")
plt.plot(test_time, y_true, label="Actual OT")
plt.plot(test_time, point_pred, label="TimesFM point forecast")
plt.fill_between(
    test_time,
    lower,
    upper,
    alpha=0.2,
    label="TimesFM P10-P90 interval",
)
plt.title("TimesFM 2.5 Zero-shot Forecast on ETTh1 OT")
plt.xlabel("Timestamp")
plt.ylabel("OT")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "timesfm_forecast.png", dpi=180)
plt.close()

# ============================================================
# 打印结果
# ============================================================
print("\n================ Metrics ================")
print(metrics.round(6).to_string(index=False))

print("\nSaved files:")
print(OUT_DIR / "metrics.csv")
print(OUT_DIR / "forecast_values.csv")
print(OUT_DIR / "timesfm_forecast.png")
