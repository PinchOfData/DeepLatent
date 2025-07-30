import torch
from torch.utils.data import Dataset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from typing import Optional, Dict
import pandas as pd

class Corpus(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        modalities: Optional[Dict[str, Dict]] = None,
        prevalence: Optional[str] = None,
        content: Optional[str] = None,
        prediction: Optional[str] = None,
        labels: Optional[str] = None,
    ):
        self.df = df
        self.modalities_config = modalities

        self.processed_modalities = {}

        for modality_name, modality_info in self.modalities_config.items():
            column = modality_info.get("column", "doc")
            views = modality_info.get("views", {})
            self.processed_modalities[modality_name] = {}

            for view_name, view_config in views.items():
                view_type = view_config["type"]
                view_column = view_config.get("column", column)

                if view_type == "bow":
                    vec = view_config.get("vectorizer", CountVectorizer())
                    if hasattr(vec, "vocabulary_"):
                        M = vec.transform(df[view_column])
                    else:
                        M = vec.fit_transform(df[view_column])
                    self.processed_modalities[modality_name][view_name] = {
                        "matrix": M,
                        "vectorizer": vec,
                        "type": "bow"
                    }

                elif view_type == "embedding":
                    embed_fn = view_config["embed_fn"]
                    texts = df[view_column].tolist()
                    M = embed_fn(texts)
                    if isinstance(M, list):
                        M = torch.stack([torch.tensor(e) for e in M])
                    elif isinstance(M, np.ndarray):
                        M = torch.tensor(M)
                    elif not isinstance(M, torch.Tensor):
                        raise TypeError("Embedding output must be list, np.ndarray, or torch.Tensor")
                    self.processed_modalities[modality_name][view_name] = {
                        "matrix": M,
                        "type": "embedding"
                    }

                elif view_type == "vote":
                    M = df[view_column].astype(float).values
                    mask = np.isnan(M)
                    M = np.nan_to_num(M)
                    self.processed_modalities[modality_name][view_name] = {
                        "matrix": M,
                        "mask": mask,
                        "type": "vote"
                    }

                elif view_type == "discrete_choice":
                    if isinstance(view_column, str):
                        view_column = [view_column]
                    self.processed_modalities[modality_name][view_name] = {}
                    for col in view_column:
                        M = dmatrix(f"~ C({col}) - 1", df)
                        self.processed_modalities[modality_name][view_name][col] = {
                            "matrix": np.asarray(M, dtype=np.float32),
                            "columns": M.design_info.column_names
                        }
                    self.processed_modalities[modality_name][view_name]["type"] = "discrete_choice"

                else:
                    raise ValueError(f"Unsupported view type: {view_type}")

        # Covariates
        self.prevalence = prevalence
        self.content = content
        self.prediction = prediction
        self.labels = labels

        self.prevalence_colnames, self.M_prevalence_covariates = (
            self._transform_df(prevalence) if prevalence else ([], np.zeros((len(df), 1), dtype=np.float32))
        )
        self.content_colnames, self.M_content_covariates = (
            self._transform_df(content) if content else ([], None)
        )
        self.prediction_colnames, self.M_prediction = (
            self._transform_df(prediction) if prediction else ([], None)
        )
        self.labels_colnames, self.M_labels = (
            self._transform_df(labels) if labels else ([], None)
        )

        self.id2token = {}
        for modality_name, views in self.processed_modalities.items():
            for view_name, info in views.items():
                if info.get("type") == "bow":
                    vocab = info["vectorizer"].get_feature_names_out()
                    self.id2token[f"{modality_name}_{view_name}"] = {
                        i: token for i, token in enumerate(vocab)
                    }

    def _transform_df(self, formula):
        M = dmatrix(formula, self.df)
        return M.design_info.column_names, np.asarray(M, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        d = {"modalities": {}}
        for modality_name, views in self.processed_modalities.items():
            d["modalities"][modality_name] = {}
            for view_name, info in views.items():
                view_type = info.get("type")
                if view_type == "bow":
                    row = info["matrix"][i]
                    row = row.toarray().squeeze(0) if scipy.sparse.issparse(row) else row
                    d["modalities"][modality_name][view_name] = torch.FloatTensor(row)
                elif view_type == "embedding":
                    d["modalities"][modality_name][view_name] = info["matrix"][i]
                elif view_type == "vote":
                    d["modalities"][modality_name][view_name] = {
                        "matrix": torch.FloatTensor(info["matrix"][i]),
                        "mask": torch.BoolTensor(info["mask"][i])
                    }
                elif view_type == "discrete_choice":
                    d["modalities"][modality_name][view_name] = {
                        col: torch.FloatTensor(info[col]["matrix"][i])
                        for col in info if col != "type"
                    }

        if self.prevalence:
            d["M_prevalence_covariates"] = self.M_prevalence_covariates[i]
        if self.content:
            d["M_content_covariates"] = self.M_content_covariates[i]
        if self.prediction:
            d["M_prediction"] = self.M_prediction[i]
        if self.labels:
            d["M_labels"] = self.M_labels[i]

        return d
