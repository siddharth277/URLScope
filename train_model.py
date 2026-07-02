import random
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from features import FEATURE_ORDER, extract_features

OUT = Path("models")
OUT.mkdir(exist_ok=True)

SAFE_DOMAINS = ["google.com","wikipedia.org","github.com","python.org","microsoft.com","apple.com","amazon.com","openai.com","stackoverflow.com","kaggle.com","coursera.org","netflix.com"]
PHISH_BRANDS = ["paypal","google","apple","amazon","bank","microsoft","instagram","wallet","crypto","netflix"]
BAD_TLDS = ["tk","ml","ga","cf","gq","top","xyz","click","zip"]

def synthesize(n=1400):
    rows = []
    labels = []
    for _ in range(n//2):
        d = random.choice(SAFE_DOMAINS)
        scheme = random.choice(["https", "https", "http"])
        path = random.choice(["", "/about", "/docs", "/products", "/search?q=help"])
        url = f"{scheme}://www.{d}{path}"
        rows.append(extract_features(url, False)); labels.append(0)
    for _ in range(n//2):
        brand = random.choice(PHISH_BRANDS)
        tld = random.choice(BAD_TLDS)
        token = random.choice(["verify", "secure", "login", "update", "confirm", "account"])
        style = random.randint(0, 4)
        if style == 0:
            url = f"http://{token}-{brand}-account.{tld}/login"
        elif style == 1:
            url = f"http://{random.randint(10,250)}.{random.randint(10,250)}.{random.randint(10,250)}.{random.randint(10,250)}/{brand}/signin"
        elif style == 2:
            url = f"http://{brand}.{token}-session-{random.randint(1000,9999)}.{tld}/verify?user={random.randint(10000,99999)}"
        elif style == 3:
            url = f"https://bit.ly/{brand}{random.randint(100,999)}"
        else:
            url = f"http://www.{brand}.com@{token}-{random.randint(100,999)}.{tld}/password-update"
        rows.append(extract_features(url, False)); labels.append(1)
    df = pd.DataFrame(rows)[FEATURE_ORDER]
    df["label"] = labels
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def build_models():
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Tree": ExtraTreeClassifier(max_depth=8, random_state=42),
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=180, max_depth=4, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42),
    }

def train(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        if "url" in df.columns and "label" in df.columns:
            features = [extract_features(u, False) for u in df["url"].astype(str)]
            X = pd.DataFrame(features)[FEATURE_ORDER]
            y = df["label"].astype(int)
        else:
            X = df[FEATURE_ORDER]
            y = df["label"].astype(int)
    else:
        df = synthesize()
        X = df[FEATURE_ORDER]
        y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42, stratify=y)
    models = build_models()
    fitted = []
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        metrics.append({"model": name, "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred), "roc_auc": roc_auc_score(y_test, prob)})
        fitted.append((name.lower().replace(" ", "_"), model))
    ensemble = VotingClassifier(estimators=fitted, voting="soft")
    ensemble.fit(X_train, y_train)
    joblib.dump({"model": ensemble, "features": FEATURE_ORDER, "metrics": pd.DataFrame(metrics)}, OUT / "urlscope_model.joblib")
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    print(train().round(4))
