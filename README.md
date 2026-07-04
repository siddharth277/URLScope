# URLScope — Intelligent Phishing URL Detection Platform

**Live App:** [https://urlscope.streamlit.app/](https://urlscope.streamlit.app/)
## Overview

URLScope is an intelligent phishing URL detection platform designed to identify malicious websites **before user interaction**. The system analyzes **lexical**, **domain**, and **behavioral** URL features using machine learning algorithms to classify URLs as **safe** or **phishing**.

The platform provides not only prediction results but also explainability using **SHAP (SHapley Additive exPlanations)**, allowing users to understand why a URL was classified as suspicious.

---

## Problem Statement

Phishing attacks are among the most common cyber threats, where attackers create fake websites to steal sensitive information such as:

* User credentials
* Banking details
* Credit card information
* Personal identity data

Traditional blacklists often fail to detect newly created phishing domains. This project solves that problem using machine learning–based detection.

---

## Features

* Real-time phishing URL detection
* Lexical URL analysis
* Domain reputation checks
* Behavioral feature extraction
* Risk confidence scoring
* SHAP-based model explainability
* Interactive web interface
* Multiple ML model comparison
* Performance-weighted voting ensemble (weak models contribute less automatically)

---

## Technologies Used

* Python 3.x
* Scikit-learn
* Pandas
* XGBoost
* SHAP
* Streamlit
* HTML / CSS

---

## Dataset

The models are trained on the **[Malicious URL Detection Dataset (Enhanced 2026)](https://www.kaggle.com/datasets/moutasmtamimi/malicious-url-detection-dataset-enhanced-2026)** from Kaggle.

* **~640,000 URLs** (`final_dataset_with_all_features_v3.1.csv`)
* Class distribution: **~67% safe / ~33% phishing** (moderately imbalanced)
* Includes both raw URLs and precomputed feature columns (character frequencies, subdomain stats, etc.), some of which are candidates for future integration into live feature extraction

---

## Machine Learning Algorithms

The following algorithms are implemented and compared:

### 1. Decision Tree

A supervised learning model that makes predictions using tree-based rules. Trained with `class_weight="balanced"` to counter class imbalance.

### 2. Random Tree (`ExtraTreeClassifier`)

An extremely randomized tree variant used as an additional, faster-training ensemble member.

### 3. Logistic Regression

A statistical classification model that predicts phishing probability.

### 4. Random Forest

An ensemble of multiple decision trees that improves prediction accuracy and reduces overfitting. Scaled up to **400 trees, max depth 16** (from 250/10) to take advantage of the larger dataset, with `n_jobs=-1` for faster training.

### 5. XGBoost

A high-performance gradient boosting model. Scaled up to **350 trees, max depth 6** (from 180/4), with `scale_pos_weight` computed automatically from the train-split class ratio to counter imbalance, and `n_jobs=-1` for faster training.

### 6. Voting Ensemble (deployed)

A soft-voting ensemble over all five models above, now **weighted by each model's own validation F1 score** (computed automatically, no manual tuning required) so weaker learners no longer drag down stronger ones. This is the model actually saved and used by the app.

---

## Handling Class Imbalance

Since the dataset is ~67% safe / ~33% phishing, plain accuracy can be misleading — a model that always predicts "safe" would still score ~67% accuracy while catching zero phishing URLs. To address this:

* `class_weight="balanced"` is applied to Decision Tree and Random Tree
* `scale_pos_weight` is applied to XGBoost, computed automatically from the actual train-split ratio
* Model comparison is judged primarily on **F1 and ROC-AUC**, not raw accuracy

---

## Feature Extraction

The system extracts three categories of features:

### Lexical Features

* URL length
* Number of dots
* Number of hyphens
* Number of digits
* Presence of `@`
* Presence of IP address
* Suspicious keywords

### Domain Features

* Domain age
* Top-level domain
* SSL certificate validity
* WHOIS data

### Behavioral Features

* Redirect count
* External resource loading
* Form action behavior
* JavaScript redirects

---

## Project Structure

```bash
urlscope/
│
├── app.py
├── features.py
├── train_model.py
├── index.html
├── style.css
├── requirements.txt
├── models/
├── data/
│   └── final_dataset_with_all_features_v3.1.csv
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/SachinGarg-hub/URLScope
cd urlscope
```

### Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Model

Run:

```bash
python train_model.py
```

This will train all machine learning models — including the weighted Voting Ensemble — and save them inside the `models/` folder. The printed metrics table now includes a **`Voting Ensemble (deployed)`** row, which is the actual model used by the app, so results can be judged correctly instead of only looking at individual base learners.

---

## Run Application

Start the application using Streamlit:

```bash
streamlit run app.py
```

---

## Workflow

1. User enters URL
2. Feature extractor processes URL
3. ML models evaluate phishing probability
4. Weighted Voting Ensemble produces the final classification
5. SHAP explains feature importance
6. Risk verdict is displayed

---

## Output Example

Input URL:

```text
https://secure-paypal-verify-account.tk/login
```

Output:

* Verdict: **Likely Phishing**
* Risk Confidence: **87%**
* Suspicious Features:

  * Suspicious TLD
  * Domain age too low
  * Invalid SSL certificate

---

## Performance Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Latest results on the 640k-row Kaggle dataset (67% safe / 33% phishing):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.7120 | 0.5433 | 0.8363 | 0.6587 | 0.8461 |
| Random Tree | 0.6695 | 0.5043 | 0.2995 | 0.3758 | 0.6156 |
| Logistic Regression | 0.7235 | 0.5779 | 0.6226 | 0.5994 | 0.7784 |
| Random Forest | 0.8476 | 0.7577 | 0.7959 | 0.7763 | 0.9256 |
| XGBoost | 0.8322 | 0.7223 | 0.8042 | 0.7611 | 0.9167 |
| **Voting Ensemble (deployed)** | 0.8359 | 0.7536 | 0.7522 | 0.7529 | 0.9065 |

Because raw accuracy is misleading under class imbalance, **F1 and ROC-AUC are the primary metrics** used to compare models. Recall improvements (e.g. XGBoost's recall rising after applying `scale_pos_weight`) directly translate to catching more real phishing URLs, even when they come at a small accuracy cost.

> Note: individual Random Forest currently edges out the unweighted ensemble on some runs, since a plain soft-voting average lets weaker base learners (Decision Tree, Random Tree) drag down stronger ones. The F1-weighted voting scheme addresses this by giving each model influence proportional to its own validation performance.

---

## Future Scope

* Integrate the ~40+ precomputed Kaggle feature columns (character frequencies, subdomain stats, etc.) that are derivable directly from the URL string, so the live sidebar scanner can compute them for brand-new URLs
* Browser extension support
* Real-time threat intelligence API
* Deep learning models
* Live WHOIS integration
* Enterprise dashboard
* Chrome extension deployment

---

## Applications

* Cybersecurity systems
* Threat intelligence
* Browser security tools
* Educational research
* Enterprise phishing defense

---

## Conclusion

URLScope provides a reliable and explainable solution for phishing detection using machine learning. By combining multiple features, class-imbalance-aware training, a performance-weighted ensemble, and explainable AI, the platform helps users identify malicious websites before becoming victims of cyber attacks.

---

## Contributors

* Sachin garg
* Goutam
* Navdeep
* Arsh kambooj
* Arnavdeep

---

## License

This project is licensed under the MIT License.
