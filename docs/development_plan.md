# Development Plan – Phase 1: Feature Engineering + Gradient-Boost Model

## Overview
This sprint adds richer engineered features and introduces a tree-based baseline (GradientBoostingClassifier) to improve UFC fight outcome predictions while formalising a train/inference pipeline.

## Work Items

### 1 Feature Engineering (`scripts/features.py`)
1. Momentum & streaks  
   - For each fighter, sort fights chronologically.  
   - Compute `win_streak` (consecutive wins prior to bout).  
   - Compute `last3_win_rate` (proportion of victories in previous 3 fights).

2. Recent-form aggregates  
   - Rolling mean over the last 3 fights for key performance metrics:  
     `sig_str_landed`, `takedowns`, `control_time`.  
   - Create both Red and Blue rolling averages then differential columns identical to existing “*_diff” pattern.

3. Physical mismatch  
   - Explicit `height_diff`, `reach_diff`, `weight_diff`, `age_diff` (age already covered).

4. Helper API changes  
   - Add `build_momentum_features(df)` and `build_recent_form_features(df)`; call them inside `engineer_features()`.  
   - Maintain a module-level list `FEATURE_COLS` of engineered numeric columns for downstream selection.

### 2 Model Refactor (`scripts/model.py`)
1. Factory interface  
   ```python
   def train_model(X, y, model_type: str = "logreg", **kwargs):
       ...
   ```
   - `logreg` → existing `LogisticRegression`.  
   - `gbdt`   → `GradientBoostingClassifier(random_state=42)`.

2. Evaluation helper  
   - `_evaluate(model, X_val, y_val)` prints Accuracy, ROC-AUC, classification report.

3. Persistence  
   - After validation, refit on full data and save as  
     `models/best.pkl` if its ROC-AUC ≥ current baseline.

### 3 Training Orchestrator (`scripts/train.py`)
CLI entry point:

```bash
python train.py --model gbdt
```

Steps:
1. Load & clean data  
   ```python
   df = clean_ufc_data("data/ufc-master.csv")
   ```
2. Feature engineering  
   ```python
   df = engineer_features(df)
   ```
3. Scaling/encoding  
   ```python
   X, y = scale_features(df)
   ```
4. Train  
   ```python
   model = train_model(X, y, model_type=args.model)
   ```

### 4 Upcoming-Fight Scoring (`scripts/predict_upcoming.py`)
- Load `models/best.pkl` and identical preprocessing pipeline.  
- Score `data/upcoming.csv`.  
- Output `predictions.csv` with columns:

| RedFighter | BlueFighter | prob_red_win | predicted_winner |
|-----------|------------|--------------|------------------|

### 5 Documentation & Requirements
- Update `docs/architecture.md` → feature-engineering bullet list.  
- Ensure `scikit-learn>=1.4` in `requirements.txt`; add if absent.

### 6 Timeline
| Day | Task |
|----|------|
| 1 | Implement `features.py` helpers + unit tests |
| 2 | Refactor `model.py` & create `train.py`; run experiments |
| 3 | Build `predict_upcoming.py` & update docs |

## Updated Flow

```mermaid
flowchart LR
    A[Clean CSV] --> B[Feature Engineering<br/>(momentum, recent)]
    B --> C[Scale/Encode]
    C --> D[Model Factory<br/>logreg / gbdt]
    D --> E[Best Model.pkl]
    E --> F[Predict Upcoming]
```

---

End of plan.