# URLSCOPE — Intelligent Phishing URL Detection Platform

This project keeps the provided `index.html` design unchanged and adds a Python/Streamlit backend for phishing URL detection.

## Tech stack
Python, Pandas, Scikit-Learn, XGBoost, SHAP, Streamlit

## Algorithms included
- Decision Tree
- Random Tree using `ExtraTreeClassifier`
- Logistic Regression
- Random Forest ensemble member
- XGBoost
- Soft-voting ensemble over all models

## Features analyzed
Lexical URL signals, domain signals, and optional behavioral/live checks:
URL length, hostname length, path length, dots, hyphens, digits, special characters, IP host, `@` trick, HTTPS, suspicious TLD, shortener domain, brand/risk keywords, subdomain count, query params, entropy, domain age, SSL validity, page reachability, forms, and external-link ratio.

## Run
```bash
cd urlscope_platform
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Optional real dataset
`train_model.py` can train from a CSV with either:

1. `url,label` columns, where label is `1` for phishing and `0` for safe.
2. Precomputed feature columns matching `features.FEATURE_ORDER` plus `label`.

Example:
```bash
python -c "from train_model import train; print(train('your_dataset.csv'))"
```

## Files
- `index.html` — original UI provided by you, preserved.
- `features.py` — lexical, domain, SSL, WHOIS, and behavioral feature extraction.
- `train_model.py` — model training and metrics.
- `app.py` — Streamlit app that renders the provided HTML UI dynamically.
- `requirements.txt` — dependencies.
