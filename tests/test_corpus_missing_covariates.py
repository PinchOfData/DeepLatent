"""Covariate formulas (prevalence, content, prediction) refuse missing values. patsy's
default drops rows with a missing value, so row i of the matrix would no longer belong to
unit i: every later unit would get its neighbour's covariates, and the last units none."""

import numpy as np
import pandas as pd
import pytest

from deeplatent import Corpus


def _corpus(df, **formulas):
    view = {"type": "embedding", "matrix": np.zeros((len(df), 2))}
    return Corpus(df, modalities={"e": {"column": "p", "views": {"w": view}}}, **formulas)


@pytest.mark.parametrize("arg", ["prevalence", "content", "prediction"])
def test_missing_covariate_raises(arg):
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "g": ["a", "b", "a"], "p": 0.0})
    with pytest.raises(Exception, match="missing values"):
        _corpus(df, **{arg: "~ x"})
    df = df.assign(x=[1.0, 2.0, 3.0], g=["a", None, "b"])
    with pytest.raises(Exception, match="missing values"):
        _corpus(df, **{arg: "~ C(g)"})


def test_complete_covariates_keep_one_row_per_unit():
    # a missing value in a column the formulas do not use is irrelevant
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "g": ["a", "b", "a"], "unused": [np.nan] * 3, "p": 0.0}
    )
    corpus = _corpus(df, prevalence="~ x + C(g)", content="~ x")
    assert corpus.prevalence_colnames == ["Intercept", "C(g)[T.b]", "x"]
    assert corpus.M_prevalence_covariates.shape == (3, 3)
    np.testing.assert_array_equal(corpus[1]["M_prevalence_covariates"], [1.0, 1.0, 2.0])
    np.testing.assert_array_equal(corpus[2]["M_content_covariates"], [1.0, 3.0])
