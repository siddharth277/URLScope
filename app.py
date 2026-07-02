import re
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from features import FEATURE_ORDER, feature_frame
from train_model import train

MODEL_PATH = Path("models/urlscope_model.joblib")
HTML_PATH = Path("index.html")

st.set_page_config(page_title="URLSCOPE", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)

def explain_with_shap(model, X):
    try:
        import shap
        # Explain the XGBoost estimator inside the ensemble when available.
        for name, estimator in model.named_estimators_.items():
            if "xgboost" in name:
                explainer = shap.TreeExplainer(estimator)
                values = explainer.shap_values(X)
                if isinstance(values, list):
                    values = values[1]
                vals = values[0]
                return pd.Series(vals, index=FEATURE_ORDER).sort_values(key=abs, ascending=False)
    except Exception:
        pass
    return pd.Series(index=FEATURE_ORDER, data=0.0)

def heuristic_reasons(feats, shap_values):
    labels = {
        "has_suspicious_tld": "Suspicious top-level domain", "brand_keyword_count": "Brand/risk keyword present",
        "domain_age_days": "Very new or unknown domain age", "ssl_valid": "No valid SSL certificate",
        "url_length": "Excessive URL length", "num_hyphens": "Hyphen count in domain",
        "has_ip": "IP address used as host", "has_at": "@ redirect trick present",
        "is_shortened": "Shortened link service", "has_forms": "Page contains forms",
        "external_link_ratio": "High external-link ratio", "has_https": "HTTPS present"
    }
    rows = []
    for f in FEATURE_ORDER:
        v = feats[f]
        bad = (f in ["has_suspicious_tld","has_ip","has_at","is_shortened","has_forms"] and v) or (f=="ssl_valid" and not v) or (f=="url_length" and v>65) or (f=="num_hyphens" and v>=2) or (f=="brand_keyword_count" and v>0) or (f=="domain_age_days" and (v == -1 or 0 <= v < 30)) or (f=="external_link_ratio" and v>0.7)
        ok = (f in ["has_ip","has_at","has_suspicious_tld","is_shortened"] and not v) or (f=="has_https" and v)
        if bad or ok:
            rows.append(("bad" if bad else "ok", labels.get(f, f.replace("_", " ").title()), str(v), float(shap_values.get(f, 0.0))))
    rows = sorted(rows, key=lambda r: abs(r[3]), reverse=True)[:8]
    return rows

def render_html(url, verdict, risk, reasons, model_name="Voting ensemble"):
    html = HTML_PATH.read_text(encoding="utf-8")
    result_title = "Flagged — likely phishing" if verdict else "Clear — likely legitimate"
    stamp = "FLAGGED<br>PHISHING" if verdict else "LIKELY<br>SAFE"
    color = "var(--red)" if verdict else "var(--teal)"
    triggered = sum(1 for r in reasons if r[0] == "bad")
    rows = "\n".join([f'''<div class="ev-row"><div class="ev-flag {flag}"></div><div class="ev-name">{name}</div><div class="ev-detail">{detail}</div><div class="ev-weight">{weight:+.2f}</div></div>''' for flag,name,detail,weight in reasons])
    html = re.sub(r'<input type="text"[^>]*>', f'<input type="text" placeholder="https://secure-paypal-verify-account.tk/login" value="{url}">', html)
    html = re.sub(r'<p class="verdict-url">.*?</p>', f'<p class="verdict-url">{url}</p>', html)
    html = re.sub(r'<p class="verdict-title">.*?</p>', f'<p class="verdict-title" style="color:{color}">{result_title}</p>', html)
    html = re.sub(r'<p class="verdict-sub">.*?</p>', f'<p class="verdict-sub">{triggered} risk signals triggered</p>', html)
    html = re.sub(r'<div class="stamp">\s*<div class="stamp-text">.*?</div>\s*</div>', f'<div class="stamp" style="border-color:{color}"><div class="stamp-text" style="color:{color}">{stamp}</div></div>', html, flags=re.S)
    html = re.sub(r'<span class="value">.*?</span>', f'<span class="value" style="color:{color}">{risk:.0%}</span>', html)
    html = re.sub(r'<div class="meter-track"><div class="meter-fill"></div></div>', f'<div class="meter-track"><div class="meter-fill" style="width:{risk*100:.1f}%"></div></div>', html)
    html = re.sub(r'<div class="evidence-head"><span class="label">Feature breakdown</span></div>.*?</div>\s*<div class="model-footer">', f'<div class="evidence-head"><span class="label">Feature breakdown</span></div>{rows}</div><div class="model-footer">', html, flags=re.S)
    html = re.sub(r'<div class="pill"><div class="sw"></div>.*?</div>', f'<div class="pill"><div class="sw"></div>{model_name}</div>', html, count=1)
    return html

bundle = load_model()
model = bundle["model"]
st.sidebar.title("URLSCOPE Controls")
url = st.sidebar.text_input("URL to scan", "https://secure-paypal-verify-account.tk/login")
live = st.sidebar.checkbox("Enable live domain / SSL / page checks", value=False)
show_table = st.sidebar.checkbox("Show raw feature table", value=False)

X, feats = feature_frame(url, live)
risk = float(model.predict_proba(X)[0, 1])
verdict = risk >= 0.50
shap_values = explain_with_shap(model, X)
reasons = heuristic_reasons(feats, shap_values)
components.html(render_html(url, verdict, risk, reasons), height=1180, scrolling=True)

if show_table:
    st.subheader("Extracted features")
    st.dataframe(X.T.rename(columns={0: "value"}))
    st.subheader("Training metrics")
    st.dataframe(bundle["metrics"].round(4))
