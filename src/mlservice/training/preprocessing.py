"""Preprocessor construction.

The preprocessor is fitted as part of the pipeline and serialized inside the
artifact. Nothing about feature handling lives in the service — that is what
guarantees training/serving parity.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlservice.artifacts.schema import FeatureSchema
from mlservice.config.training import PreprocessingConfig


def build_preprocessor(schema: FeatureSchema, config: PreprocessingConfig) -> ColumnTransformer:
    """Assemble a ColumnTransformer from the frozen feature schema.

    Driving column selection from the schema (rather than re-sniffing dtypes at
    fit time) means the artifact's declared contract and its actual behaviour
    cannot diverge.
    """
    numeric = [f.name for f in schema.features if f.kind == "numeric"]
    categorical = [f.name for f in schema.features if f.kind == "categorical"]
    boolean = [f.name for f in schema.features if f.kind == "boolean"]

    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy=config.numeric_imputation))
    ]
    if config.scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    encoder_kwargs: dict[str, object] = {
        "handle_unknown": "infrequent_if_exist",
        "sparse_output": False,
    }
    if config.one_hot_min_frequency is not None:
        encoder_kwargs["min_frequency"] = config.one_hot_min_frequency

    categorical_steps = [
        (
            "impute",
            SimpleImputer(
                strategy=config.categorical_imputation,
                fill_value="__missing__" if config.categorical_imputation == "constant" else None,
            ),
        ),
        ("encode", OneHotEncoder(**encoder_kwargs)),
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical))
    if boolean:
        transformers.append(
            (
                "boolean",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent"))]),
                boolean,
            )
        )

    if not transformers:
        raise ValueError("Feature schema produced no usable columns")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_features(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Reorder and coerce a raw frame to the schema, without fitting anything."""
    return schema.validate_frame(frame)
