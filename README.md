# mm_features_extractor

This repository contains lightweight tools and analysis notebooks to extract and analyze market-making features, with a focus on order-book (LOB) snapshots, trade execution logs and inventory metrics. The code and sample CSVs are tailored for a market-making workflow (Bitget & Binance data) used for feature engineering and exploratory analysis.

**Key Points:**
- **Purpose:** collect LOB snapshots and extract features for ML, and analyze market-making performance metrics.
- **Primary scripts:** `binance_lob_crawler.py` (LOB collector), `Bitget_mm_pynb.ipynb` (analysis notebook).

**Repository Structure**
- `binance_lob_crawler.py`: a Redis-backed LOB collector that reads ticker and orderbook keys and writes daily CSV files. Useful to run as a lightweight daemon to capture LOB features (best bid/ask, notional depth, last price) at ~1s intervals.
- `Bitget_mm_pynb.ipynb`: Jupyter notebook containing EDA and visualizations for trade logs (`mm_bitget.csv`, futures logs, inventory snapshots). Includes inventory reconstruction, VWAP/spread capture, fee tracking, and PnL calculations.
- `mm_bitget.csv`, `mm_bitget_futures.csv`, `cake_inventory.csv`, and other `*.csv` files: sample data used by the notebook.

**Important Notes**
- `binance_lob_crawler.py` uses a local Redis instance (`localhost:6379`) by default; adjust the connection parameters if your Redis runs elsewhere.
- The collector writes daily CSV files into a configurable directory (default in the script). Ensure the process has write permission.
- The notebook contains data-cleaning and calculation steps that assume specific CSV column names (see the notebook cells). Review the first code cell for required columns and date parsing.

**Contributing / Extending**
- Add more LOB-derived features (e.g., imbalance, slope, PCA of depth) in `extract_lob_features`.
- Add tests or a small CLI wrapper if you plan to run collectors across multiple symbols.

**License & Contact**
- This repo contains user scripts and example data. Add a license file if you intend to share or distribute broadly.
- For questions, contact the repository owner.

**Data privacy / Publishing policy**
- **Do not publish raw CSV files.** The repository may contain CSV files with trade logs, inventory snapshots or other potentially sensitive market data — these should never be committed to a public repository.
- To ensure CSV files are not tracked going forward, a `.gitignore` has been added to ignore `*.csv` files.