# SST AI Swingtrade & Adaptive ML Stock Scanner

An adaptive, quantitative swing-trading scanner built for **SwingStockTraders (SST)**.

## Key Features
- **Adaptive Stock Scoring**: Automatically categorizes stocks (AI/Software, Semiconductors, Quantum, Biotech, Commodity/Mining) and adjusts component weights accordingly.
- **Genuine Machine Learning**: TimeSeriesSplit cross-validated CatBoost/RandomForest probabilities trained on technical features with zero look-ahead bias.
- **Multi-Timeframe Alignment**: Synchronizes 1D, 1H, and 15M trend indicators.
- **Strict Data Integrity**: Returns `DATA UNAVAILABLE` when metrics cannot be reliably retrieved rather than fabricating scores.

## Local Installation & Run

```bash
git clone [https://github.com/YOUR_USERNAME/sst-ai-scanner.git](https://github.com/YOUR_USERNAME/sst-ai-scanner.git)
cd sst-ai-scanner
pip install -r requirements.txt
streamlit run app.py
