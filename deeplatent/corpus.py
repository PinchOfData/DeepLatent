import torch
from torch.utils.data import Dataset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
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
        # Per-view availability: which rows (units) have the modality at all. A view's
        # "available" option (a boolean column name or a boolean array of length N)
        # marks the rows without it; their entries are placeholders that the models
        # never read. This differs from missing values inside a modality (e.g. a vote
        # not cast), which the modality's own likelihood handles.
        self.availability = {}

        for modality_name, modality_info in self.modalities_config.items():
            column = modality_info.get("column", "doc")
            views = modality_info.get("views", {})
            self.processed_modalities[modality_name] = {}

            for view_name, view_config in views.items():
                view_type = view_config["type"]
                view_column = view_config.get("column", column)
                key = f"{modality_name}_{view_name}"
                avail = (
                    self._availability_vector(view_config["available"], df, key)
                    if "available" in view_config
                    else None
                )

                if view_type == "bow":
                    if "matrix" in view_config:
                        # Precomputed term counts (units x vocabulary, scipy.sparse),
                        # for corpora tokenized outside the package; "vocab"
                        # (optional) names the columns.
                        M = view_config["matrix"]
                        if not scipy.sparse.issparse(M):
                            raise TypeError(
                                "A precomputed bow matrix must be scipy.sparse "
                                "(units x vocabulary term counts)."
                            )
                        M = scipy.sparse.csr_matrix(M)
                        vocab = view_config.get("vocab")
                        if vocab is not None and len(vocab) != M.shape[1]:
                            raise ValueError(
                                f"vocab has {len(vocab)} terms but the bow matrix has "
                                f"{M.shape[1]} columns."
                            )
                        self.processed_modalities[modality_name][view_name] = {
                            "matrix": M,
                            "vocab": vocab,
                            "type": "bow",
                        }
                    else:
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
                    if "matrix" in view_config:
                        # Precomputed votes as a scipy.sparse matrix (+1 yea, -1 nay, 0 not
                        # observed), for long panels where each unit sees a small share of the
                        # roll calls; rows are expanded one at a time in __getitem__.
                        M = view_config["matrix"]
                        if not scipy.sparse.issparse(M):
                            raise TypeError(
                                "A precomputed vote matrix must be scipy.sparse "
                                "(+1 yea, -1 nay, 0 not observed)."
                            )
                        # Explicitly stored zeros are "not observed", like implicit
                        # ones.
                        M = scipy.sparse.csr_matrix(M, copy=True)
                        M.eliminate_zeros()
                        self.processed_modalities[modality_name][view_name] = {
                            "matrix": M,
                            "type": "vote",
                        }
                    else:
                        M = df[view_column].astype(float).values
                        mask = np.isnan(M)
                        M = np.where(mask, 2.0, M)
                        # float32 tensors once, so that __getitem__ returns row views
                        self.processed_modalities[modality_name][view_name] = {
                            "matrix": torch.from_numpy(M.astype(np.float32)),
                            "mask": torch.from_numpy(mask),
                            "type": "vote",
                        }

                elif view_type == "discrete_choice":
                    if isinstance(view_column, str):
                        view_column = [view_column]
                    answers = df
                    if avail is not None and avail.any() and not avail.all():
                        # Absent rows are placeholders, never read: give them an
                        # observed answer so that the design keeps one row per unit.
                        absent = ~avail.numpy()
                        answers = df[view_column].copy()
                        first = int(np.flatnonzero(~absent)[0])
                        for col in view_column:
                            answers.loc[absent, col] = df[col].iloc[first]
                    self.processed_modalities[modality_name][view_name] = {}
                    for col in view_column:
                        # NA_action="raise": patsy would otherwise drop rows with a
                        # missing answer and misalign every later unit.
                        M = dmatrix(f"~ C({col}) - 1", answers, NA_action="raise")
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
                            "available": None if avail is None else avail.numpy(),
                            "type": "image"
                        }

                else:
                    raise ValueError(f"Unsupported view type: {view_type}")

                if avail is not None:
                    self.availability[key] = avail

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
                    if "vectorizer" in info:
                        vocab = info["vectorizer"].get_feature_names_out()
                    else:
                        vocab = info["vocab"] or [
                            str(j) for j in range(info["matrix"].shape[1])
                        ]
                    self.id2token[f"{modality_name}_{view_name}"] = {
                        i: token for i, token in enumerate(vocab)
                    }

    @staticmethod
    def _availability_vector(spec, df, key):
        if isinstance(spec, str):
            if not pd.api.types.is_bool_dtype(df[spec]):
                raise TypeError(
                    f"'available' column '{spec}' for '{key}' must be boolean."
                )
            a = df[spec].to_numpy(dtype=bool)
        else:
            a = np.asarray(spec)
        if a.dtype != np.bool_ or a.shape != (len(df),):
            raise TypeError(
                f"'available' for '{key}' must be a boolean column name or a boolean "
                f"array of length {len(df)}."
            )
        return torch.from_numpy(a.copy())

    @staticmethod
    def _expand_votes(S):
        """Sparse vote rows (+1 yea, -1 nay, 0 not observed) -> the dense coding of the
        vote view: matrix 1 yea, 0 nay, 2 if missing; mask True if missing."""
        S = scipy.sparse.csr_matrix(S)
        rows = torch.from_numpy(np.repeat(np.arange(S.shape[0]), np.diff(S.indptr)))
        cols = torch.from_numpy(S.indices.astype(np.int64))
        matrix = torch.full(S.shape, 2.0)
        matrix[rows, cols] = torch.from_numpy((S.data > 0).astype(np.float32))
        mask = torch.ones(S.shape, dtype=torch.bool)
        mask[rows, cols] = False
        return {"matrix": matrix, "mask": mask}

    @staticmethod
    def _load_image(info, i):
        """Lazy loading: transform a single image on demand. A row without the
        modality gets a zero placeholder shaped like the view's images (never read)."""
        avail = info.get("available")
        if avail is not None and not avail[i]:
            if "placeholder" not in info:
                present = np.flatnonzero(avail)
                info["placeholder"] = (
                    torch.zeros_like(Corpus._load_image(info, int(present[0])))
                    if len(present)
                    else torch.zeros(3, 224, 224)
                )
            return info["placeholder"]
        image_path = info["image_paths"][i]
        transform_fn = info["transform_fn"]
        try:
            # Transform function should handle single image path -> tensor
            tensor = transform_fn([image_path])  # Pass as list for consistency
            if isinstance(tensor, list):
                tensor = tensor[0]  # Extract single tensor
            elif isinstance(tensor, torch.Tensor) and tensor.dim() == 4:
                tensor = tensor[0]  # Remove batch dimension if present
            return tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback: create zero tensor with reasonable default shape
            return torch.zeros(3, 224, 224)

    def _transform_df(self, formula):
        # NA_action="raise": patsy would otherwise drop rows with a missing value, so
        # that row i of the matrix no longer belongs to unit i (every later unit gets
        # its neighbour's covariates). Missing covariates must be handled explicitly.
        M = dmatrix(formula, self.df, NA_action="raise")
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
        if not (np.isscalar(i) or (torch.is_tensor(i) and i.dim() == 0)):
            return self.get_batch(i)
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
                    if scipy.sparse.issparse(info["matrix"]):
                        # Same coding as the dense path: 1 yea, 0 nay, 2 and masked if missing.
                        votes = self._expand_votes(info["matrix"][i])
                        d["modalities"][modality_name][view_name] = {
                            k: v[0] for k, v in votes.items()
                        }
                    else:
                        d["modalities"][modality_name][view_name] = {
                            "matrix": info["matrix"][i],
                            "mask": info["mask"][i],
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
                        image = self._load_image(info, i)
                        d["modalities"][modality_name][view_name] = image

        if self.prevalence:
            d["M_prevalence_covariates"] = self.M_prevalence_covariates[i]
        if self.content:
            d["M_content_covariates"] = self.M_content_covariates[i]
        if self.prediction:
            d["M_prediction"] = self.M_prediction[i]
        if self.labels_info:
            d["M_labels"] = self.M_labels[i]
        if self.availability:
            d["available"] = {k: v[i] for k, v in self.availability.items()}

        return d

    def get_batch(self, indices):
        """The items at `indices`, already collated (what a DataLoader with the default
        collate function yields for them), built a whole batch at a time: each matrix is
        sliced once instead of row by row. corpus[list_of_indices] calls this."""
        idx = np.asarray(indices, dtype=np.int64)
        t = torch.from_numpy(idx)
        d = {"modalities": {}}
        for modality_name, views in self.processed_modalities.items():
            d["modalities"][modality_name] = {}
            for view_name, info in views.items():
                view_type = info.get("type")
                if view_type == "bow":
                    rows = info["matrix"][idx]
                    if scipy.sparse.issparse(rows):
                        rows = rows.toarray()
                    out = torch.FloatTensor(np.asarray(rows))
                elif view_type == "embedding":
                    out = info["matrix"][t]
                elif view_type == "vote":
                    if scipy.sparse.issparse(info["matrix"]):
                        out = self._expand_votes(info["matrix"][idx])
                    else:
                        out = {"matrix": info["matrix"][t], "mask": info["mask"][t]}
                elif view_type == "discrete_choice":
                    out = {
                        col: torch.FloatTensor(info[col]["matrix"][idx])
                        for col in info if col != "type"
                    }
                elif view_type == "image":
                    if "matrix" in info:
                        out = info["matrix"][t]
                    else:
                        out = torch.stack([self._load_image(info, i) for i in idx])
                d["modalities"][modality_name][view_name] = out

        if self.prevalence:
            prevalence = self.M_prevalence_covariates[idx]
            d["M_prevalence_covariates"] = torch.from_numpy(prevalence)
        if self.content:
            d["M_content_covariates"] = torch.from_numpy(self.M_content_covariates[idx])
        if self.prediction:
            d["M_prediction"] = torch.from_numpy(self.M_prediction[idx])
        if self.labels_info:
            d["M_labels"] = torch.from_numpy(self.M_labels[idx])
        if self.availability:
            d["available"] = {k: v[t] for k, v in self.availability.items()}
        return d
