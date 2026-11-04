# Orchestra Subscription Prediction

**Classical Music Meets Classical ML — Fall 2026**  
Predicting which former orchestra patrons will purchase a subscription for the **2014–15 concert season**.

[Kaggle competition](https://www.kaggle.com/competitions/classical-music-meets-classical-ml-fall-2026)

## Project overview

An orchestra wants to identify previous patrons most likely to purchase an upcoming season subscription. Using historical subscription purchases, individual ticket purchases, account information, donations, and residential geography, we developed an account-level binary classification model that estimates each patron's probability of subscribing.

The competition evaluates **area under the ROC curve (AUROC)**. Our submission contains continuous probabilities rather than hard 0/1 predictions. Because the private leaderboard uses a separate portion of the test set, model selection was based on cross-validation of the labeled training data—not repeated optimization against the public leaderboard.

### Results at a glance

| Item | Result |
| --- | ---: |
| Labeled training accounts | 6,941 |
| Training subscribers | 332 (4.78%) |
| Unlabeled Kaggle test accounts | 2,975 |
| Final predictive features | 39 |
| Selected algorithm | CatBoost |
| Five-fold CV, mean AUROC across three seeds | **0.97055** |
| Averaged out-of-fold AUROC, 15-model ensemble | **0.97180** |
| Kaggle public AUROC | **0.97615** |
| Public leaderboard position | **10th** (snapshot: November 2, 2026) |

**Important:** The public score is computed from approximately half of the Kaggle test set. The private score and final ranking are not known from this result. AUROC measures *ranking quality*, not classification accuracy or probability calibration.

## Data

The competition provides the following CSV files. Place them under `data/raw/` after downloading them from Kaggle.

| File | Role |
| --- | --- |
| `train.csv` | Account IDs and 2014–15 subscription labels (`0`/`1`) |
| `test.csv` | Account IDs to predict |
| `sample_submission.csv` | Required Kaggle output schema and account order (`ID,Predicted`) |
| `account.csv` | Patron account attributes, addresses, and donation information |
| `subscriptions.csv` | Historical season subscriptions |
| `tickets_all.csv` | Historical individual ticket purchases |
| `zipcodes.csv` | ZIP-level location attributes |
| `concerts.csv` | Historical concert information, explored during EDA |
| `concerts_2014-15.csv` | Planned upcoming concerts, explored as possible context |

The final baseline feature pipeline uses `train`, `test`, `account`, `subscriptions`, `tickets_all`, and `zipcodes`. The two concert files were supplied and explored but are **not direct inputs** to the final 39-feature dataset. Raw data and generated account-level predictions are intentionally kept outside version control.

## Approach

### 1. Exploratory data analysis

We inspected each source dataset, verified identifiers and duplicates, quantified feature missingness, and explored the imbalanced target. Only **332 of 6,941 labeled patrons** purchased the target-season subscription, making AUROC more appropriate than raw accuracy. Average precision was tracked as a secondary metric.

### 2. Feature engineering

`src/features.py` joins the source tables into **one row per account** and creates 39 predictive features in four groups:

| Feature group | Examples |
| --- | --- |
| Subscription behavior | Prior subscribed seasons, last subscription year, recency, consecutive renewal streak, package, tier, seats |
| Ticket behavior | Ticket counts, seats, unique concert sets, recent purchases, ticket recency |
| Geography | Billing ZIP, city, state, latitude, longitude |
| Account and donations | Account attributes, 2013 donation amount and donation indicator |

Historical ticket and subscription records are restricted to seasons **before 2014–15**. Account IDs and target labels are excluded from model inputs. Potentially post-cutoff *lifetime* donation totals are excluded by default. Missing count-based history is represented as zero when appropriate; unavailable recency remains missing, alongside explicit history indicators.

We separately evaluated additional geographic demographics, venue distances, and neighborhood-density signals. These experiments helped test specific hypotheses without changing the baseline dataset used for the final submission.

### 3. Validation and model comparison

We used **five-fold stratified cross-validation** on labeled *accounts*, repeated with random seeds `42`, `123`, and `2026`. This is **not** a chronological train/validation split: all labels refer to subscription decisions for the same target season, and predictors summarize prior seasons. Preprocessing learned from data is fitted within training folds, and the Kaggle test labels are never accessed.

Six model families were benchmarked: CatBoost, XGBoost, LightGBM, Random Forest, Extra Trees, and a PyTorch multilayer perceptron. We also evaluated a logistic regression baseline, Optuna-tuned boosting models, and predetermined probability/rank ensembles.

| Candidate | Mean OOF AUROC (three seeds) |
| --- | ---: |
| **CatBoost — original parameters** | **0.97055** |
| CatBoost — tuned | 0.97037 |
| CatBoost + XGBoost probability mean | 0.96908 |
| XGBoost — tuned | 0.96660 |
| LightGBM — tuned | 0.96537 |
| Random Forest | 0.96388 |
| Multilayer perceptron | 0.96382 |
| Extra Trees | 0.96112 |

The original CatBoost configuration was the strongest overall and proved more reliable than the tested blends. It uses **650 iterations, learning rate 0.035, tree depth 5, and L2 leaf regularization 5**. The tuning score used to choose hyperparameters is not treated as an independent holdout result.

### 4. Feature ablation and geography experiments

Feature ablation demonstrated that both historical subscriptions and geography are valuable:

| Feature set | Features | Mean OOF AUROC |
| --- | ---: | ---: |
| **All original features** | **39** | **0.97055** |
| Subscriptions + geography | 26 | 0.96742 |
| No geography | 34 | 0.96237 |
| Subscriptions only | 21 | 0.95355 |

Removing geographic features lowered AUROC by approximately **0.00818**. In the final model's CatBoost importance analysis, `zip_lat`, `zip_long`, and recent subscription indicators were among the highest-ranked inputs.

Explicit venue-distance features initially improved one random seed, but a more reliable three-seed comparison favored the original 39 features (**0.97055** vs. **0.96941** mean AUROC). We therefore did not include the additional distance features in the final model.

### 5. Final prediction strategy

`src/predict.py` trains the selected CatBoost configuration on **five folds for each of three random seeds** and averages the resulting **15 test-prediction vectors**. Each individual model is fitted on approximately 80% of the labeled accounts; the ensemble combines their probabilities for the 2,975 unseen Kaggle accounts. The script also produces out-of-fold diagnostics and checks that the submission exactly matches the official sample's IDs, column names, and row order.

The averaged out-of-fold prediction achieved **0.97180 AUROC**, compared with the original mean seed AUROC of **0.97055**. This remains an internal validation result, not a separate untouched holdout evaluation.

## Repository structure

```text
orchestra-subscription-prediction/
├── README.md
├── requirements-modeling.txt
├── requirements-advanced.txt
├── data/
│   ├── raw/                    # Kaggle source CSVs (not committed)
│   └── processed/              # Feature tables, experiments, final CSV (not committed)
├── src/
│   ├── features.py             # Reproducible account-level feature engineering
│   ├── geography.py            # Optional geographic feature blocks
│   ├── evaluate.py             # Fold-safe model validation
│   ├── train.py                # Initial model baseline and submission format
│   ├── experiments.py          # Feature ablation and multi-seed studies
│   ├── geography_experiments.py
│   ├── models.py               # Six model families and preprocessing
│   ├── model_search.py         # Benchmarking and Optuna tuning
│   ├── ensemble.py             # Fixed prediction blends
│   └── predict.py              # Final 15-model CatBoost ensemble
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering_review.ipynb
│   ├── 03_model_validation_review.ipynb
│   ├── 04_model_experiments.ipynb
│   ├── 05_geography_feature_review.ipynb
│   ├── 06_model_comparison.ipynb
│   └── 07_final_model_review.ipynb
├── tests/                       # Pipeline, preprocessing and submission checks
└── docs/                        # Detailed experiment and submission notes
```

The exact files under `data/processed/` are generated by the scripts; they do not need to be checked into Git.

## Reproducing the submission

Run these commands from the repository root in an Ubuntu/WSL terminal. Python **3.10 or newer** is recommended.

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-advanced.txt
```

If a compatible virtual environment already exists, activate it instead of recreating it. Download the competition CSVs into `data/raw/`, preserving their original filenames.

### 2. Build the baseline features

```bash
python -m src.features
```

This creates `data/processed/train_features.csv` and `data/processed/test_features.csv`, with identical predictive columns and one record per account.

### 3. Run tests and generate the final Kaggle CSV

```bash
python -m unittest discover -s tests -v
python -m src.predict
```

The final submission is written to:

```text
data/processed/final/catboost_final.csv
```

Its required schema is:

```csv
ID,Predicted
```

The `Predicted` column contains a probability between 0 and 1 for every test ID. Upload **only this CSV** using Kaggle's *Submit Prediction* interface.

### 4. Review results in notebooks

Open the notebooks in VS Code with the project's Python/Jupyter kernel and run them in numerical order as needed. Each notebook explains a distinct stage of the workflow. For the final review, run `notebooks/07_final_model_review.ipynb` after `src.predict`. Its last code cell verifies submission IDs, row order, probability bounds, and agreement with the generated test predictions.

### Optional: reproduce the model comparison

The main modeling experiments can be rerun with:

```bash
python -m src.model_search --mode benchmark
python -m src.model_search --mode tune --models catboost xgboost lightgbm --trials 15
python -m src.model_search --mode compare --seeds 42 123 2026
```

These commands take longer than generating a single prediction file. Their results are saved under `data/processed/model_search/` and can be inspected in Notebook 06. The experiment reports are useful for auditing model selection; they are not required to generate the final submission.

## Reproducibility, leakage prevention, and limitations

- **Point-in-time features:** Subscription and ticket histories are restricted to seasons beginning before 2014. Lifetime donation totals are excluded because their snapshot date is uncertain.
- **Separation of identifiers and predictors:** `account.id` and the target `label` never enter the model input matrix.
- **Cross-validation hygiene:** Model evaluation uses held-out account folds; model-dependent preprocessing is fitted inside the relevant training data.
- **Class imbalance:** Only 4.78% of the labeled accounts are positive. AUROC is the competition metric; average precision provides an additional perspective.
- **Model-selection uncertainty:** Extensive experimentation on the same labeled dataset can introduce selection optimism. Multi-seed CV improves stability assessment but is not a substitute for a fully independent holdout.
- **Public versus private leaderboard:** The 0.97615 public score is informational. The private score may differ; the public test set was not used to select features, models, or blend weights.
- **Prediction interpretation:** AUROC evaluates relative ranking, not calibrated purchase probabilities. Feature importance describes model behavior rather than causal drivers.

Before publishing the repository, review `.gitignore` and notebook outputs for customer-level identifiers. Commit source code, documentation, tests, and appropriately sanitized notebooks—not raw patron records or generated account-level predictions.

## Documentation and acknowledgments

- [Model optimization and comparison](docs/model_optimization.md)
- [Geographic feature engineering](docs/geographic_feature_engineering.md)
- [Feature ablation experiments](docs/feature_ablation.md)
- [Final submission guide](docs/final_submission.md)
- Kaggle competition: *Classical Music Meets Classical ML Fall 2026*, Jon Reifschneider, 2026.

This repository was developed for a Duke University machine learning course project. The competition data and evaluation environment are provided by the course organizer via Kaggle.
