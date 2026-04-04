# Dataset balancing for glucose prediction (Loop + ai_ready and beyond)

This document consolidates guidance on combining the Loop dataset with ai_ready-style data, balancing for machine learning, and aligning training with real-world CGM-only / missing-covariate use.

---

## 1. Combining Loop with ai_ready (glucose-only from healthy cohort)

### 1.1 What you are combining

| Dataset | Population | Notes |
|--------|------------|--------|
| **Loop** | Type 1 diabetes on automated insulin delivery (AID) | Rich context: glucose, basal, bolus, carbs; wide glucose range |
| **ai_ready** (glucose only) | Mostly healthy / metabolically normal | Often no insulin; internal subgroups by health condition |

### 1.2 Is the combined dataset balanced?

**No** — imbalance appears on several axes at once.

1. **Condition imbalance**  
   Loop is T1D AID; ai_ready is largely healthy. Physiology differs: insulin-driven swings vs tight endogenous regulation. A naive union lets the larger source dominate and mixes conflicting dynamics.

2. **Glucose distribution skew**  
   Healthy CGM clusters in ~70–140 mg/dL. T1D has fatter tails (hypo, hyper). Raw mixing underweights clinically important extremes unless you correct for it.

3. **Internal groups within ai_ready**  
   If ai_ready has subgroups (e.g. healthy, pre-diabetic, impaired tolerance), the largest subgroup dominates unless you stratify.

4. **Sequence length and density**  
   Loop sequences can be long and dense; other cohorts may use shorter windows. Unequal sequence counts change what the model “sees” per user type.

5. **Feature asymmetry**  
   Loop has insulin/carbs/basal; ai_ready (glucose-only) does not. A multivariate model trained mainly on Loop cannot use the same feature vector for the healthy side without a deliberate missing-data design.

---

## 2. How to balance the combined dataset

1. **Stratify by condition group**  
   Label sequences by health condition (T1D, pre-diabetic, healthy, etc.). Do not treat “healthy” as one undifferentiated block.

2. **User-level quota sampling**  
   Cap contribution per group (e.g. target fractions across T1D / pre-diabetic / healthy). Sample **users** first, then sequences, to reduce leakage between train/validation/test.

3. **Glucose-range stratification within groups**  
   Preserve each group’s distribution while ensuring rare but important bands (e.g. &lt;70, &gt;250 mg/dL) are not drowned out — e.g. histogram-aware or weighted sampling.

4. **Sequence-length normalization**  
   Use fixed-length windows (e.g. 4–8 hours) from all sources so training is not dominated by one cohort’s typical sequence length.

5. **Loss weighting**  
   Use per-sample or per-bin weights inversely related to frequency of the target glucose range so extremes are not ignored.

---

## 3. Additional data sources (better coverage for prediction)

**Bridge between healthy and T1D**

- Pre-diabetic / impaired glucose tolerance cohorts with CGM (fills the gap between “healthy” and “T1D AID”).
- Public T1D datasets with rich context (e.g. OhioT1DM: CGM + pump + meals + activity/sleep/HR — small but benchmarked).
- Larger open pump/CGM resources where licensing allows (e.g. community donations) — check terms of use.

**Generalization beyond AID**

- T1D on MDI (manual therapy) — different from pump/AID dynamics.
- T2D on oral meds or basal-only — different insulin physiology than classic T1D AID.

**Feature enrichment (when you add modalities)**

- Activity / steps / HR — strong drivers of short-horizon glucose.
- Meal timing and composition — even simple carb logs add signal; Loop already motivates this for the T1D side.

**Practical priority**  
One of the largest gains for a **prediction** model is **balanced representation of hypoglycemic segments** (&lt;70 mg/dL): they are rare in healthy data, more common in T1D, and clinically critical. Without oversampling or loss weighting, models often underfit these regions.

---

## 4. Real-world use: unknown user, CGM-first, optional insulin/carbs

When the model is deployed on real CGM exports, you typically do **not** know whether the user is diabetic or healthy, and insulin/carbs may be **missing because the user forgot to log**, not because nothing happened.

### 4.1 Input contract

- Treat insulin and carbs as **optional**.
- Distinguish **unknown / not logged** from **explicit zero**:
  - Prefer explicit missingness flags (e.g. `insulin_logged`, `carbs_logged`) rather than zero-filling nulls (zero-fill implies “no dose,” which is wrong when data are missing).
- Train with **random masking** of insulin/carb inputs so glucose-only prediction is a first-class mode, not a broken fallback.

### 4.2 What the model can infer without a user label

A sufficiently long **glucose history** (e.g. 2–4 hours, or more for personalization) encodes regime: healthy vs dysregulated vs partially controlled patterns differ in variance, baseline, and recovery. The model does not need a diagnosis field if the architecture uses history as the main signal.

### 4.3 Personalization without “which user” in the API

- **Latent user embedding** from recent CGM (or a short warm-up period) to capture individual dynamics.
- **Light online adaptation** or meta-learning so new users are handled after a few days of data.

### 4.4 Uncertainty

When covariates are missing, error grows — especially at longer horizons. Prefer **probabilistic outputs** (intervals, quantiles, calibrated uncertainty) and let uncertainty **widen** when insulin/carbs are unlogged, matching what you simulate in training.

### 4.5 Training data that matches deployment

| Mix | Purpose |
|-----|--------|
| Loop with full covariates | Learn effects when logging is complete |
| Loop with covariates **masked** | Mimic “diabetic who forgot to log” |
| Healthy / ai_ready glucose-only | Healthy regime without pump fields |
| Short windows | New users before much history exists |
| Long histories | Stable personalization signal |

### 4.6 Horizon vs available data

Rough expectation:

| Horizon | Glucose-only | With reliable insulin/carbs |
|--------|--------------|-----------------------------|
| ~15 min | Strong for many users | Small gain |
| ~30 min | Good for healthy; moderate for T1D | Meaningful gain for T1D |
| ~60 min | Weaker for T1D without context | Large gain when context is present |

Consider **multi-horizon predictions** with **horizon-specific uncertainty**, so the product stays honest when only CGM is available.

---

## 5. Summary checklist

- [ ] Stratify by health condition; quota at **user** level.
- [ ] Weight or sample for **hypo/hyper** bands, not only overall row counts.
- [ ] Align sequence construction (e.g. fixed windows) across sources.
- [ ] Never equate **missing** insulin/carbs with **zero**; use flags + masked training.
- [ ] Use **history-heavy** inputs and optional **personalization** for unknown users.
- [ ] Output **uncertainty**; widen it under missing covariates.
- [ ] Add **bridge cohorts** (pre-diabetes) and, over time, **diverse therapy types** for robustness.

---

*This document reflects planning notes for combining Loop-processed data with other CGM cohorts for glucose forecasting. Update it as dataset definitions and product constraints evolve.*
