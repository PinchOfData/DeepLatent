import torch
from torch.utils.data import Dataset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from typing import Optional, Dict, Union
import pandas as pd

class Corpus(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        modalities: Optional[Dict[str, Dict]] = None,
        prevalence: Optional[str] = None,
        content: Optional[str] = None,
        prediction: Optional[str] = None,
        labels: Optional[Dict[str, Dict]] = None,
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
                    if "matrix" in view_config:
                        M = view_config["matrix"]
                        if isinstance(M, np.ndarray):
                            M = torch.tensor(M)
                        elif isinstance(M, list):
                            M = torch.stack([torch.tensor(e) for e in M])
                        elif not isinstance(M, torch.Tensor):
                            raise TypeError("Provided embedding matrix must be list, np.ndarray, or torch.Tensor")
                    else:
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
                    M = np.where(mask, 2.0, M)
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

                elif view_type == "image":
                    if "matrix" in view_config:
                        M = view_config["matrix"]
                        if isinstance(M, np.ndarray):
                            M = torch.tensor(M)
                        elif isinstance(M, list):
                            M = torch.stack([torch.tensor(img) for img in M])
                        elif not isinstance(M, torch.Tensor):
                            raise TypeError("Provided image matrix must be list, np.ndarray, or torch.Tensor")
                        self.processed_modalities[modality_name][view_name] = {
                            "matrix": M,
                            "type": "image"
                        }
                    else:
                        # Store paths and transform function for lazy loading
                        image_paths = df[view_column].tolist()
                        transform_fn = view_config.get("transform_fn", None)
                        if transform_fn is None:
                            raise ValueError("Image view type requires a 'transform_fn' parameter")
                        
                        self.processed_modalities[modality_name][view_name] = {
                            "image_paths": image_paths,
                            "transform_fn": transform_fn,
                            "type": "image"
                        }

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
        self.labels_info, self.M_labels = (
            self._process_labels(labels) if labels else ({}, None)
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

    def _process_labels(self, labels_config: Dict[str, Dict]):
        """
        Process dict-based labels configuration.

        Args:
            labels_config: Dict mapping label names to their config, e.g.:
                {
                    "sentiment": {"column": "sentiment_score", "type": "regression"},
                    "category": {"column": "category_id", "type": "multiclass", "num_classes": 5},
                    "is_spam": {"column": "spam_flag", "type": "binary"}
                }

        Returns:
            labels_info: Dict with metadata per label (type, num_classes, start_idx, end_idx, column)
            M_labels: np.ndarray with all label values concatenated
        """
        labels_info = {}
        label_arrays = []
        current_idx = 0

        for label_name, config in labels_config.items():
            column = config["column"]
            label_type = config["type"]

            if label_type not in {"regression", "binary", "multiclass"}:
                raise ValueError(f"Invalid label type '{label_type}' for label '{label_name}'. "
                               f"Must be 'regression', 'binary', or 'multiclass'.")

            values = self.df[column].values

            if label_type == "regression":
                # Continuous values
                arr = values.astype(np.float32).reshape(-1, 1)
                num_classes = None
                end_idx = current_idx + 1
            elif label_type == "binary":
                # Binary 0/1 values
                arr = values.astype(np.float32).reshape(-1, 1)
                num_classes = None
                end_idx = current_idx + 1
            elif label_type == "multiclass":
                # Integer class indices
                num_classes = config.get("num_classes")
                if num_classes is None:
                    num_classes = int(values.max()) + 1
                arr = values.astype(np.int64).reshape(-1, 1)
                end_idx = current_idx + 1

            labels_info[label_name] = {
                "type": label_type,
                "num_classes": num_classes,
                "start_idx": current_idx,
                "end_idx": end_idx,
                "column": column
            }

            label_arrays.append(arr)
            current_idx = end_idx

        if label_arrays:
            M_labels = np.concatenate(label_arrays, axis=1).astype(np.float32)
        else:
            M_labels = None

        return labels_info, M_labels

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
                elif view_type == "image":
                    if "matrix" in info:
                        d["modalities"][modality_name][view_name] = info["matrix"][i]
                    else:
                        # Lazy loading: transform single image on demand
                        image_path = info["image_paths"][i]
                        transform_fn = info["transform_fn"]
                        
                        try:
                            # Transform function should handle single image path -> tensor
                            tensor = transform_fn([image_path])  # Pass as list for consistency
                            if isinstance(tensor, list):
                                tensor = tensor[0]  # Extract single tensor
                            elif isinstance(tensor, torch.Tensor) and tensor.dim() == 4:
                                tensor = tensor[0]  # Remove batch dimension if present
                            d["modalities"][modality_name][view_name] = tensor
                        except Exception as e:
                            print(f"Error loading image {image_path}: {e}")
                            # Fallback: create zero tensor with reasonable default shape
                            d["modalities"][modality_name][view_name] = torch.zeros(3, 224, 224)

        if self.prevalence:
            d["M_prevalence_covariates"] = self.M_prevalence_covariates[i]
        if self.content:
            d["M_content_covariates"] = self.M_content_covariates[i]
        if self.prediction:
            d["M_prediction"] = self.M_prediction[i]
        if self.labels_info:
            d["M_labels"] = self.M_labels[i]

        return d
