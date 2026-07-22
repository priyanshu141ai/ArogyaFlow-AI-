from collections.abc import Sequence
from typing import Any

import numpy as np
import shap  # type: ignore[import-untyped]

from arogyaflow.exceptions import TrainingDataError


def shap_summary(
    model: Any,
    transformed: np.ndarray[Any, Any],
    feature_names: Sequence[str],
    sample_size: int,
) -> list[dict[str, object]]:
    sample = np.asarray(transformed)[:sample_size]
    names = [str(name) for name in feature_names]
    if not len(sample) or not names:
        raise TrainingDataError("SHAP explanation requires features and samples")
    if hasattr(model, "coef_"):
        values = np.asarray(shap.LinearExplainer(model, sample)(sample).values)
    else:
        values = np.asarray(shap.TreeExplainer(model).shap_values(sample, check_additivity=False))
    if values.shape[-1] == len(names):
        feature_axis = values.ndim - 1
    else:
        feature_axis = next(
            (index for index, size in enumerate(values.shape) if size == len(names)), -1
        )
    if feature_axis < 0:
        raise TrainingDataError("SHAP output does not match transformed features")
    importance = np.abs(np.moveaxis(values, feature_axis, -1)).reshape(-1, len(names)).mean(axis=0)
    ranking = sorted(zip(names, importance, strict=True), key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_absolute_shap": float(value)} for name, value in ranking[:20]]
