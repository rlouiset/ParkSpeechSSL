from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

IndividualKey = Tuple[str, str]  # (dataset_name, patient_hash), same as data.IndividualKey

NUMERIC_FIELDS = ("age", "updrs", "hy", "years_dx")


def load_metadata(path: Path) -> Dict[IndividualKey, dict]:
    """metadata.csv (see preprocessing_scripts/build_metadata.py) -> {(dataset, patient_hash): row},
    numeric fields parsed to float (nan when unknown), sex left as "M"/"F"/""."""
    metadata = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for field in NUMERIC_FIELDS:
                row[field] = float(row[field]) if row[field] else math.nan
            metadata[(row["dataset"], row["patient_hash"])] = row
    return metadata


def effective_rank(x: np.ndarray) -> float:
    """exp(entropy) of the normalized singular values of the centred matrix (Roy & Vetterli, 2007)."""
    x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum()))


def _within_corpus(y: np.ndarray, datasets: List[str]) -> np.ndarray:
    """Subtracts each corpus' mean target, so a probe can't score by recognising the corpus
    (e.g. KCL's PD patients have higher H&Y than NeuroVoz's) -- only within-corpus
    differences between individuals count."""
    datasets = np.asarray(datasets)
    y = y.copy()
    for corpus in np.unique(datasets):
        y[datasets == corpus] -= y[datasets == corpus].mean()
    return y


def _summary(name: str, fold_scores: List[float], pooled: float, n: int) -> Dict[str, float]:
    return {
        f"{name}": pooled,
        f"{name}_fold_mean": float(np.nanmean(fold_scores)),
        f"{name}_fold_std": float(np.nanstd(fold_scores)),
        f"{name}_n": float(n),
    }


def cv_regression(embd: np.ndarray, datasets: List[str], y: np.ndarray, n_splits: int, seed: int, name: str):
    """Ridge on one embedding per individual, so K-fold is subject-level, on the corpus-centred
    target. Scored by Spearman rho of out-of-fold predictions, pooled over folds and per fold
    (per-fold rho on ~10 subjects is noisy -- its std is the spread to compare variants against)."""
    x, y = embd, _within_corpus(y, datasets)
    oof = np.zeros_like(y)
    fold_scores = []
    for train_idx, test_idx in KFold(n_splits, shuffle=True, random_state=seed).split(x):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        model.fit(x[train_idx], y[train_idx])
        oof[test_idx] = model.predict(x[test_idx])
        fold_scores.append(spearmanr(oof[test_idx], y[test_idx]).statistic)
    return _summary(name, fold_scores, spearmanr(oof, y).statistic, len(y))


def cv_classification(x: np.ndarray, y: np.ndarray, n_splits: int, seed: int, name: str):
    """Logistic regression, subject-level stratified K-fold, scored by AUC."""
    oof = np.zeros(len(y))
    fold_scores = []
    for train_idx, test_idx in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(x, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(x[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        fold_scores.append(roc_auc_score(y[test_idx], oof[test_idx]))
    return _summary(name, fold_scores, roc_auc_score(y, oof), len(y))


def run_clinical_probes(
    keys: List[IndividualKey],
    labels: List[str],
    embd: np.ndarray,
    metadata: Dict[IndividualKey, dict],
    n_splits: int,
    seed: int,
) -> Dict[str, float]:
    """One (segment-averaged) embedding per individual in, metrics out. Every probe is
    within one class, to measure how much intra-class variability the embedding keeps:
    - erank_{pd,hc}: effective rank of all PD / HC individuals' embeddings (metadata not needed)
    - updrs_pd: NeuroVoz total UPDRS
    - hy_pd: H&Y (NeuroVoz + KCL)
    - age_{pd,hc}: NeuroVoz, IPVS, FredPrior. Never pooled across classes, since IPVS'
      young-HC group makes age a proxy for diagnosis there
    - sex_{pd,hc}: AUC
    """
    labels_np = np.array(labels)
    rows = [metadata.get(k) for k in keys]
    datasets = np.array([k[0] for k in keys])
    sex = np.array([r["sex"] if r is not None else "" for r in rows])
    results = {}

    def field(name: str) -> np.ndarray:
        return np.array([r[name] if r is not None else math.nan for r in rows])

    for label in ("PD", "HC"):
        suffix = label.lower()
        in_class = labels_np == label
        results[f"erank_{suffix}"] = effective_rank(embd[in_class])

        targets = (("updrs", "hy", "age") if label == "PD" else ("age",))
        for target in targets:
            y = field(target)
            mask = in_class & ~np.isnan(y)
            if mask.sum() >= 2 * n_splits:
                results.update(
                    cv_regression(embd[mask], list(datasets[mask]), y[mask], n_splits, seed, f"{target}_{suffix}")
                )

        mask = in_class & np.isin(sex, ["M", "F"])
        if min((sex[mask] == "M").sum(), (sex[mask] == "F").sum()) >= n_splits:
            results.update(cv_classification(embd[mask], (sex[mask] == "M").astype(int), n_splits, seed, f"sex_{suffix}"))
    return results
