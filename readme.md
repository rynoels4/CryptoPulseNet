# PulseBTC

**Fast, minute-level Bitcoin price prediction** — a clean, well-engineered PyTorch pipeline for fetching minute data, feature engineering, hybrid LSTM+Transformer modeling, and walk‑forward validation.

---

<p align="center">
  <em>Predict the next minute — safely, reproducibly, and with clear artifacts.</em>
</p>

---

## Quick links
- **Status:** Prototype
- **Language:** Python 3.9+
- **Frameworks:** PyTorch, pandas, TA‑Lib

---

## Features
- Minute-level BTC data fetching with Parquet caching for resumable downloads.
- Robust feature engineering: TA indicators, time-of-day cyclical features, microstructure aggregation (optional).
- Sliding-window dataset and memory-efficient dataloaders.
- Hybrid model: LSTM backbone + Transformer encoder head (configurable).
- Walk‑forward validation with safe periodic retraining and automatic rollback on degradation.
- Checkpointing, scaler persistence, and result plotting.

---

## Getting started
Clone and install dependencies:

```bash
git clone https://github.com/rynoels4/pulsebtc.git
cd pulsebtc
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (optional — only for live fetching):

```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

If you prefer to work offline, place a `btc_1m_data.parquet` file in the project root and the pipeline will use that instead of fetching.

---

## Configuration
All runtime defaults live in a `Config` class in the main script. You can edit values such as:
- `SYMBOL` (default: `BTCUSDT`)
- `INTERVAL` (default: `1m`)
- `N_STEPS`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`
- `CACHE_FILE`, `CHECKPOINT_DIR`, `SCALER_X_PATH`, `SCALER_Y_PATH`

You can also override `Config` programmatically if you run the pipeline from another script.

---

## Usage
Run the end-to-end pipeline (fetch → features → train → walk-forward → plots):

```bash
python CryptoPulseNet.py
```

---

## Outputs & artifacts
- `btc_1m_data.parquet` — cached raw data
- `checkpoint_best.pt`, `checkpoint_last.pt` — saved model checkpoints
- `final_model_pytorch.pt` — final exported model (configurable)
- `scaler_x.joblib`, `scaler_y.joblib` — saved scalers
- `prediction_results.png` — evaluation plot

---

## Model & training notes
- Mixed precision and `torch.compile()` are conditionally enabled when supported.
- Gradient accumulation is supported for large effective batch sizes on limited GPU RAM.
- Retraining can be head-only (freeze backbone) to reduce catastrophic forgetting and speed up updates.

---

## Tips
- Install TA‑Lib system dependencies first if you use TA features.
- Match your `torch` wheel to your CUDA runtime to avoid compatibility issues.
- Keep scalers if you plan to run inference later — they are required to transform inputs consistently.

---

## Contributing
Contributions welcome! Please:
1. Open an issue for feature requests or bug reports.
2. Fork → create a feature branch → add tests & docs → open a pull request.

Please keep changes backwards compatible and document new config options in this README.

---

## License
MIT — include a `LICENSE` file in the repo if you want this to be the official license.

---

## Contact
If you want help customizing this README, adding CI badging, or generating a `requirements.txt`, ping me here and I’ll prepare it.

*Generated for a high-frequency Bitcoin prediction project.*

