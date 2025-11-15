"""Utilities for constructing sparse hedging weights against a COG index."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _prepare_target(
    index_cog: pd.Series | dict | Iterable[tuple[str, float]],
    target_columns: Sequence[str],
) -> np.ndarray:
    """Return the target vector as an ndarray in the expected column order."""
    series = pd.Series(index_cog)
    missing = [col for col in target_columns if col not in series.index]
    if missing:
        raise ValueError(f"index_cog is missing required keys: {missing}")
    return series.loc[list(target_columns)].astype(float).to_numpy()


def _validate_dataframe(
    df: pd.DataFrame,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    """Ensure the dataframe provides the required columns."""
    missing = [col for col in target_columns if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    return df


def _resolve_target_columns(
    df: pd.DataFrame,
    index_cog: pd.Series | dict,
    target_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    """Determine the ordered list of target columns to use for matching."""
    if target_columns is None:
        index_like = getattr(index_cog, "index", index_cog)
        index_keys = set(index_like)
        candidate_columns = [col for col in df.columns if col in index_keys]
        if not candidate_columns:
            raise ValueError(
                "Unable to infer target columns: no overlap between df and index_cog."
            )
        return tuple(candidate_columns)

    return tuple(target_columns)


def find_min_abs_weights(
    df: pd.DataFrame,
    index_cog: pd.Series | dict,
    *,
    target_columns: Sequence[str] | None = None,
    combination_size: int | None = None,
    tolerance: float = 1e-9,
) -> pd.Series:
    """Return the sparsest three-position hedge with the smallest L1 norm.

    Parameters
    ----------
    df:
        DataFrame containing instrument exposures. Must include the columns in
        ``target_columns``.
    index_cog:
        Target exposure, supplied as a Series or dict with ``target_columns`` keys.
    target_columns:
        Iterable of exposure column names to match. Defaults to the columns that
        are present in both ``df`` and ``index_cog`` (order taken from ``df``).
    combination_size:
        Number of instruments to include in each candidate combination. If not
        provided, defaults to the number of ``target_columns`` so that the
        system is square. Set explicitly (e.g. to 3) to enforce a specific
        sparsity level.
    tolerance:
        Numerical tolerance for accepting a solution (in exposure units).

    Returns
    -------
    pd.Series
        Weights indexed by the instrument labels for the optimal three-position
        combination.

    Examples
    --------
    >>> weights = find_min_abs_weights(df, index_cog, combination_size=3)
    >>> float(weights.sum())  # doctest: +SKIP
    0.0

    Raises
    ------
    ValueError
        If no combination of three instruments can match the target within the
        specified tolerance.
    """

    target_columns = _resolve_target_columns(df, index_cog, target_columns)

    df = _validate_dataframe(df, target_columns)
    target = _prepare_target(index_cog, target_columns)

    combo_size = combination_size or len(target_columns)
    if combo_size <= 0:
        raise ValueError("combination_size must be a positive integer.")
    if combo_size > len(df.index):
        raise ValueError("combination_size exceeds the number of available rows.")

    best_score = np.inf
    best_solution: tuple[tuple[object, ...], np.ndarray] | None = None

    for combo in itertools.combinations(df.index, combo_size):
        exposures = df.loc[list(combo), target_columns].to_numpy(dtype=float).T

        if np.linalg.matrix_rank(exposures) < min(exposures.shape):
            continue

        weights, residuals, _, _ = np.linalg.lstsq(exposures, target, rcond=None)

        residual = exposures @ weights - target
        if np.linalg.norm(residual, ord=np.inf) > tolerance:
            continue

        score = np.sum(np.abs(weights))
        if score < best_score:
            best_score = score
            best_solution = (combo, weights)

    if best_solution is None:
        raise ValueError("No valid three-instrument combination matched the target.")

    combo, weights = best_solution
    return pd.Series(weights, index=pd.Index(combo, name="instrument"))


def fast_find_min_abs_weights(
    df: pd.DataFrame,
    index_cog: pd.Series | dict,
    *,
    target_columns: Sequence[str] | None = None,
    combination_size: int | None = None,
    tolerance: float = 1e-9,
) -> pd.Series:
    """Optimized variant that precomputes exposure matrices for faster iteration.

    Examples
    --------
    >>> weights = fast_find_min_abs_weights(df, index_cog, combination_size=3)
    >>> float(weights.sum())  # doctest: +SKIP
    0.0
    """

    target_columns = _resolve_target_columns(df, index_cog, target_columns)
    df = _validate_dataframe(df, target_columns)
    target = _prepare_target(index_cog, target_columns)

    combo_size = combination_size or len(target_columns)
    if combo_size <= 0:
        raise ValueError("combination_size must be a positive integer.")

    labels = df.index.to_list()
    if combo_size > len(labels):
        raise ValueError("combination_size exceeds the number of available rows.")

    exposures_all = df.loc[:, target_columns].to_numpy(dtype=float, copy=False)
    best_score = np.inf
    best_solution: tuple[tuple[object, ...], np.ndarray] | None = None

    for combo in itertools.combinations(range(len(labels)), combo_size):
        submatrix = exposures_all[list(combo)].T

        if np.linalg.matrix_rank(submatrix) < min(submatrix.shape):
            continue

        weights, _, _, _ = np.linalg.lstsq(submatrix, target, rcond=None)

        residual = submatrix @ weights - target
        if np.linalg.norm(residual, ord=np.inf) > tolerance:
            continue

        score = np.sum(np.abs(weights))
        if score < best_score:
            best_score = score
            best_solution = (tuple(labels[idx] for idx in combo), weights)

    if best_solution is None:
        raise ValueError("No valid three-instrument combination matched the target.")

    combo, weights = best_solution
    return pd.Series(weights, index=pd.Index(combo, name="instrument"))


__all__ = ["find_min_abs_weights", "fast_find_min_abs_weights"]
