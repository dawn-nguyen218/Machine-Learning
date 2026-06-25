# Machine-Learning

# Sustainable Concrete Mix Design Tool

A Streamlit web app that predicts concrete compressive strength and embodied
CO₂ from a mix design, flags inputs outside the training range, and suggests
comparable-strength mixes with a smaller carbon footprint.

It reproduces the deployed model from our report: a **Random Forest (200 trees,
`random_state=42`)** trained on the 8 raw mix variables plus 6 engineered
features (`binder`, `w_c_ratio`, `w_b_ratio`, `scm_fraction`,
`agg_binder_ratio`, `log_age`), fit on the 70% training partition of the
cleaned UCI Concrete Compressive Strength dataset (1,005 unique mixes after
duplicate removal). Held-out test RMSE ≈ 5.26 MPa, R² = 0.89. Lower-carbon
alternatives are searched over the full dataset.

> **Design-exploration / screening aid only.** Predictions are estimates from a
> single public dataset and do **not** replace laboratory testing, building-code
> compliance, or professional engineering judgment. Validate any mix
> experimentally before use in construction.

## Files
- `app.py` — the Streamlit application
- `requirements.txt` — dependencies
- `concrete.csv` — offline copy of the UCI dataset (used only if the live
  `ucimlrepo` fetch is unavailable; same column order as repository id 165)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens in your browser (default http://localhost:8501). Enter a mix in
the sidebar, set a target strength and curing age, and click **Run prediction**.

## Deploy (Streamlit Community Cloud)
1. Push `app.py`, `requirements.txt`, and `concrete.csv` to a GitHub repo.
2. At share.streamlit.io, create a new app pointing at `app.py` on that repo.
3. Deploy — Streamlit installs `requirements.txt` automatically.

## Data
On startup the app pulls the dataset directly from the UCI ML Repository via
`ucimlrepo` (Concrete Compressive Strength, id 165; Yeh, 1998). If that fetch
fails (e.g. no internet), it falls back to the bundled `concrete.csv`. Either
way it removes 25 exact-duplicate rows, leaving 1,005 unique mixes, then
computes embodied CO₂ from published LCA factors before training.

The Colab notebook remains the full analysis record (EDA, baselines, model
comparison, grouped-split robustness check, error analysis); this app is the
interactive front end for the final model.
