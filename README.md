## VentureGuard AI – Venture Risk Intelligence

VentureGuard AI is a multi-page **Streamlit** application that turns simple startup funding data into an investor-style **failure risk signal**.

The goal is to provide a compact, transparent decision-support layer for **VCs, LPs, and founders** – something you can realistically use in an IC meeting, not a toy demo.

---

### 1. Product Overview

- **Problem**: Early-stage venture decisions are often made with noisy benchmarks and fragmented data. Even understanding whether a company's funding profile is thin or robust relative to history requires manual work.
- **Solution**: VentureGuard AI uses a trained machine learning model on historical startup data to estimate the probability that a startup will **not** reach a sustained operating or acquired outcome, given its funding profile.
- **Scope**: This version focuses **narrowly** on numeric funding features:
  - `funding_total_usd` – total capital raised.
  - `funding_rounds` – number of distinct funding rounds.

It is designed as an **additive risk lens** for investment discussions, not a replacement for qualitative diligence.

---

### 2. Application Features

- **Multi-page Streamlit UI**
  - **Home** – context, dataset overview, and usage guidance.
  - **Risk Assessment** – main scoring interface for individual startups.
  - **Analytics** – portfolio-scale visuals derived from the historical dataset.
  - **Model Insights** – explanation of model design, feature importance, and coverage.
  - **About** – methodology, stack, caveats, and potential extensions.

- **Risk Assessment**
  - Clean form for:
    - Total funding raised (USD, in millions).
    - Number of funding rounds.
  - Outputs:
    - **Failure Probability (%)**.
    - **VC-style Risk Score (0–100)**.
    - **Risk Tier**: Low / Medium / High Risk.
    - Plain-English explanation comparing the startup to dataset medians.

- **Analytics**
  - Median funding vs. outcome (operating/acquired vs closed/other).
  - Distribution of funding rounds across companies.
  - Outcome mix (success vs failure proportions) with counts and shares.

- **Model Insights**
  - High-level model description (RandomForest classifier with SMOTE).
  - Feature importance for the numeric features used.
  - Dataset coverage snapshot (e.g. by country).
  - Clear articulation of assumptions and blind spots.

---

### 3. Tech Stack

- **Frontend / App**: [Streamlit](https://streamlit.io/)
- **Modeling**: [scikit-learn](https://scikit-learn.org/) `RandomForestClassifier`
- **Data**: Historical startup funding dataset (`big_startup_secsees_dataset.csv`)
- **Language**: Python 3.10+

Key Python dependencies are listed in `requirements.txt`.

---

### 4. Project Structure

```text
VentureGaurd AI/
├─ app.py                  # Home page entrypoint for Streamlit
├─ core.py                 # Shared utilities: paths, loading, inference, explanations
├─ pages/
│  ├─ 1_Risk_Assessment.py # Main scoring interface
│  ├─ 2_Analytics.py       # Dataset analytics and distributions
│  ├─ 3_Model_Insights.py  # Model explanation and feature importance
│  └─ 4_About.py           # About, methodology, and caveats
├─ requirements.txt        # Python dependencies
└─ README.md               # This file
```

The trained model (`ventureguard_model.pkl`) and dataset (`big_startup_secsees_dataset.csv`) are expected to live **one level above** the project folder (e.g. `C:\\Users\\HP`), mirroring the original notebook setup.

If you move them into the project directory, update `MODEL_PATH` and `DATA_PATH` in `core.py` accordingly.

---

### 5. Setup & Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-org-or-user>/ventureguard-ai.git
cd ventureguard-ai
```

2. **Ensure data and model files are available**

Place the following files **one directory above** the project root (or adjust paths in `core.py` if you prefer a different layout):

- `big_startup_secsees_dataset.csv`
- `ventureguard_model.pkl`

3. **Create a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.\.venv\Scripts\activate         # Windows
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Run the application**

```bash
streamlit run app.py
```

Open the provided local URL in your browser to access VentureGuard AI.

---

### 6. Page-by-Page Guide

#### Home (`app.py`)

- Introduces the problem and the VentureGuard AI approach.
- Shows dataset-level metrics:
  - Number of companies.
  - Survival/exit rate.
  - Median funding and funding rounds.
- Explains how to use the other pages in an investment workflow.

#### Risk Assessment (`pages/1_Risk_Assessment.py`)

- Accepts:
  - Total funding raised (USD, in millions).
  - Number of funding rounds.
- Computes:
  - Failure probability based on the model’s `predict_proba`.
  - VC-style risk score (0–100).
  - Risk tier (Low / Medium / High).
- Provides:
  - Plain-English explanation referencing dataset medians.
  - Guidance on how to interpret the score in IC / deal triage contexts.

#### Analytics (`pages/2_Analytics.py`)

- Visualises:
  - Median funding by outcome class.
  - Distribution of funding rounds.
  - Outcome proportions for success vs failure.
- Built with Altair for clear, decision-oriented charts.

#### Model Insights (`pages/3_Model_Insights.py`)

- Describes:
  - RandomForest model design and target label.
  - Features used and class imbalance handling.
- Shows:
  - Feature importances (when available).
  - High-level data coverage by geography.
- Details:
  - Assumptions, blind spots, and safe-usage guidelines.

#### About (`pages/4_About.py`)

- Summarises:
  - Product positioning and use cases.
  - Technical stack and methodology.
  - Limitations and future extension ideas.

---

### 7. Model & Data Details

- **Target variable**:
  - `success = 1` for companies with `status` in `["operating", "acquired"]`.
  - `success = 0` otherwise.

- **Features**:
  - `funding_total_usd`:
    - Cleaned by replacing `"-"` with missing values and converting to numeric.
    - Missing values imputed with the median.
  - `funding_rounds`:
    - Treated as an integer count of rounds.
    - Missing values imputed with the median.

- **Model**:
  - `RandomForestClassifier(n_estimators=100, random_state=42)`
  - Trained on a SMOTE-resampled training set to mitigate class imbalance.

- **Prediction semantics**:
  - The model natively predicts **success probability**.
  - VentureGuard AI inverts this to expose a **failure probability** and a risk score.

---

### 8. Limitations & Responsible Use

This application is intentionally conservative about its scope.

- It only considers:
  - Total funding raised.
  - Number of funding rounds.
- It **does not** see:
  - Team quality, founder track record, or investor syndicate strength.
  - Market structure, competition, macro regime changes.
  - Unit economics, retention, or customer quality.
  - Round structure (inside rounds, flat/down rounds, secondary, etc.).

**Do not** use VentureGuard AI as an automated investment decision engine.

Recommended usage:

- As a **signal for risk sensing and triage**, especially across large portfolios.
- In combination with:
  - IC memos.
  - Board and founder insight.
  - Deeper market and product analysis.

---

### 9. Roadmap / Future Work

Potential directions:

- Add richer features:
  - Time between rounds.
  - Investor identity and syndicate quality.
  - Sector / geography-specific signals.
- Calibrate scores against actual fund outcomes (TVPI / DPI) rather than just company survival.
- Wrap the model behind an API and integrate it with:
  - CRM / deal-flow tools.
  - Portfolio dashboards.

---

### 10. Contributing

Contributions are welcome. Ideas that are particularly helpful:

- Improving model calibration or adding new, interpretable features.
- Enhancing the UI for fund-level portfolio views.
- Adding tests and CI workflows for model loading and data validation.

If you open a PR, please include:

- A short description of the change and its motivation.
- Before/after screenshots for UI changes, where relevant.

---

### 11. License

Choose the license that best fits your intended usage (for example, MIT or Apache 2.0) and add a `LICENSE` file to the repository.

Until then, treat this project as **All Rights Reserved** by default.

