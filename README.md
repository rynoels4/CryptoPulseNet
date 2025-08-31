High-Frequency Bitcoin Price Prediction (PyTorch)

A PyTorch implementation for minute-level Bitcoin prediction with strong engineering for feature creation, sliding-window datasets, hybrid LSTM+Transformer model, walk-forward validation with safe periodic retraining, and optional microstructure / RL components. The implementation includes Binance data fetching (with Parquet caching), TA-Lib indicator engineering, careful scaling to avoid leakage, and utilities for checkpointing and evaluation. 

Table of contents

Features

Requirements

Quickstart

Configuration

Usage examples

Model & training details

Artifacts & outputs

Tips & notes

Contributing

License

Features

.env support (python-dotenv) for keys & config. 

Binance data fetch with pagination + Parquet cache to resume-downloads. 

TA-Lib based feature engineering (multiple indicators + time-of-day cyclic features). 

Sliding-window dataset with configurable stride (memory efficient). 

Hybrid modelling: LSTM feature extractor + Transformer encoder head (multitask optional). 

Walk-forward validation with safe periodic retrain and optional reversion if retrain degrades validation. 

Optional microstructure streaming (Binance websocket) aggregator for L2/trade features. 

Optional RL predictor/trading modules. 

Requirements

Create a virtual environment and install dependencies (example requirements.txt below):

python>=3.9
numpy
pandas
matplotlib
python-dotenv
joblib
ta-lib
binance-connector
torch
scikit-learn


(Adjust versions for your environment / CUDA capability.)

Quickstart

Clone the repo and enter it.

Create a .env file with your Binance API keys (if you want to fetch live data):

BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here


Install dependencies:

python -m venv venv
source venv/bin/activate    # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt


Run the main script (the provided script exposes a main() that orchestrates fetch → features → train → walk-forward → plot). Example:

python path/to/your_script.py


This will:

fetch/cached minute data from Binance (Parquet cache file),

engineer features,

split into train/test,

scale features/targets,

train the configured model,

run walk-forward validation and produce plots/metrics. 

If you prefer to only use cached data, put a btc_1m_data.parquet file in the repo or edit Config.CACHE_FILE.

Configuration

All runtime defaults live in the Config class (edit or override before running). Notable defaults:

SYMBOL = 'BTCUSDT', INTERVAL = '1m', DATA_START_DATE = '2022-01-01', CACHE_FILE = 'btc_1m_data.parquet'. 

Model/training: N_STEPS, BATCH_SIZE, EPOCHS, LEARNING_RATE, GRAD_ACCUM_STEPS etc. 

Hybrid model toggles: USE_HYBRID, LSTM_*, TR_* params for Transformer sizes. 

Walk-forward & retrain: RETRAIN_FREQUENCY, RETRAIN_HEAD_ONLY, WALK_FORWARD_MAX_STEPS and safe defaults. 

Artifact paths: FINAL_MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH, CHECKPOINT_BEST_PATH, etc. 

You can either edit the Config class in the script or subclass/replace it when programmatically running the pipeline.

Usage examples
Fetch & cache data only
from your_script import Config, DataFetcher

cfg = Config()
fetcher = DataFetcher(cfg)
df = fetcher.fetch_and_cache()


This function will resume from the last timestamp in the cache if present. 

Run the full pipeline (script main()):
python your_script.py


Outputs: metrics printed to console, prediction_results_pytorch.png saved, and model/checkpoints saved per config. 

Start microstructure stream (optional)

If you have binance-connector websocket and want L2/trade features:

from your_script import start_microstructure_stream, Config
cfg = Config()
start_microstructure_stream(cfg)


This collects minute-aggregated microstructure features into MICROSTRUCTURE_FILE. 

Model & training details

Implemented models: LSTMRegressor (simple LSTM head) and HybridLSTMTransformerRegressor (LSTM → projection → TransformerEncoder → MLP head). Both support optional multitask auxiliary classification.

Training uses AMP (mixed precision) when GPU available and supports gradient accumulation to reduce VRAM footprint. Scheduler options: onecycle, cosine, plateau. Early stopping and EMA/SWA options exist. 

Walk-forward logic: trains on initial history, then iteratively predicts on df_test while optionally performing safe retrains every RETRAIN_FREQUENCY steps. Retrains can freeze backbone (heads only) and revert weights if validation degrades. 

Artifacts & outputs

Parquet cache: btc_1m_data.parquet (default). 

Checkpoints: checkpoint_best.pt, checkpoint_last.pt (configurable). 

Final model: final_model_pytorch.pt (configurable). 

Results plot saved as prediction_results_pytorch.png (cumulative log returns). 

Tips & notes

TA-Lib must be installed at system level on many OSes (e.g., apt install -y libta-lib0 / build from source) before pip install ta-lib.

For local development without Binance keys, create or reuse a btc_1m_data.parquet file and skip live fetching. 

If using CUDA, ensure torch matches your CUDA runtime to gain speedups; script can try torch.compile() for PyTorch 2.x if enabled in config. 

The project saves scalers (SCALER_X_PATH / SCALER_Y_PATH) — keep those if you plan to do inference later. 

Contributing

Open an issue for bugs or feature requests.

For PRs, please:

add tests where relevant,

document new config options in this README,

keep changes backwards compatible where possible.

License

Add your preferred license (e.g., MIT). If you want, I can append a short MIT license block to this README.
