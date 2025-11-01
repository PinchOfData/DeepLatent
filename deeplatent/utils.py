#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
import warnings
from typing import List, Optional
from tqdm import tqdm
import spacy
from spacy.cli import download as spacy_download
import torch
import re 


class text_processor:
    """
    Clean a corpus with SpaCy.
    """

    def __init__(
        self,
        spacy_model: str,
        remove_punctuation: bool = True,
        remove_digits: bool = True,
        stop_words: List[str] = [],
        lowercase: bool = True,
        lemmatize: bool = True,
        pos_tags_to_keep: Optional[List[str]] = None,
        remove_chars: Optional[List[str]] = None,
    ):
        """
        Args:
            spacy_model : string, name of the SpaCy model to use
            remove_punctuation : boolean, whether to remove punctuation or not
            remove_digits : boolean, whether to remove digits or not
            stop_words : list of strings, list of stop words to remove
            lowercase : boolean, whether to lowercase or not
            lemmatize : boolean, whether to lemmatize or not
            pos_tags_to_keep : list of strings, list of Part of Speech tags to keep
            remove_chars : list of strings, list of characters to remove
        """

        if not spacy.util.is_package(spacy_model):
            spacy_download(spacy_model)

        self.spacy_model = spacy_model
        self._nlp = spacy.load(spacy_model, disable=["ner", "parser"])
        self._nlp.add_pipe("sentencizer")
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.pos_tags_to_keep = pos_tags_to_keep
        self.remove_chars = remove_chars

    def clean_text(self, s) -> List[str]:
        """
        Clean a SpaCy doc.
        """

        if self.remove_punctuation:
            s = [t for t in s if t.is_punct == False]

        if self.remove_digits:
            #s = [t for t in s if t.is_digit == False]
            s = [t for t in s if not re.search(r'\d', t.text)]
        
        if self.pos_tags_to_keep is not None:
            temp = []
            for t in s:
                if t.pos_ in self.pos_tags_to_keep:
                    temp.append(t)
            s = temp

        if self.lowercase and not self.lemmatize:
            s = [t.lower_ for t in s]

        if self.lowercase and self.lemmatize:
            s = [t.lemma_.lower() for t in s]

        if not self.lowercase and not self.lemmatize:
            s = [t.text for t in s]

        if self.remove_chars is not None:
            for char in self.remove_chars:
                s = [t.replace(char, " ") for t in s]

        s = [t for t in s if t not in self.stop_words]

        s = [t.strip() for t in s if t not in self.stop_words]

        s = " ".join(s)

        s = s.strip()

        return s

    def process_docs(
        self,
        docs: List[str],
        batch_size: int = 100,
        n_process: int = -1,
        output_path: Optional[str] = None,
        return_docs: bool = False,
        reload_model_every=10000,  # helps with memory leakage issues
        progress_bar: bool = True,
    ):
        """
        SpaCy pipeline with multiprocessing.
        """

        spacy_docs = self._nlp.pipe(docs, batch_size=batch_size, n_process=n_process)

        if progress_bar:
            spacy_docs = tqdm(spacy_docs, total=len(docs))

        counter = 0

        if output_path is None:
            docs_clean = []
            for doc in spacy_docs:
                cleaned_text = self.clean_text(doc)
                docs_clean.append(cleaned_text)
                counter += 1
                if counter > reload_model_every:
                    self._nlp = spacy.load(self.spacy_model, disable=["ner", "parser"])
                    counter = 0
        else:
            with open(output_path, "w", newline="") as csvfile:
                fieldnames = ["doc_clean"]
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                for doc in spacy_docs:
                    cleaned_text = self.clean_text(doc)
                    writer.writerow([cleaned_text])
                    counter += 1
                    if counter > reload_model_every:
                        self._nlp = spacy.load(
                            self.spacy_model, disable=["ner", "parser"]
                        )
                        counter = 0
            if return_docs:
                docs_clean = pd.read_csv(output_path)["doc_clean"].tolist()
            else:
                docs_clean = None

        return docs_clean


def check_max_local_length(max_seq_length, texts):
    max_local_length = np.max([len(t.split()) for t in texts])
    if max_local_length > max_seq_length:
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            f"the longest document in your collection has {max_local_length} words, the model instead "
            f"truncates to {max_seq_length} tokens."
        )


def bert_embeddings_from_list(
    texts, sbert_model_to_load, batch_size, max_seq_length, device="cpu"
):
    """
    Creates SBERT Embeddings from a list
    """

    model = SentenceTransformer(sbert_model_to_load, device=device)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    check_max_local_length(max_seq_length, texts)

    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def compute_mmd_loss(x, y, device, kernel = 'multiscale'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "diffusion"
    """

    if kernel == "multiscale":

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)
        
        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                    torch.zeros(xx.shape).to(device),
                    torch.zeros(xx.shape).to(device))

        bandwidth_range = [0.01, 0.1, 0.3, 0.5, 0.7, 1]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
        
        mmd_loss = torch.mean(XX + YY - 2. * XY)
            
    if kernel == "diffusion":

        eps=1e-6
        n, d = x.shape
        t_values = [0.05,0.1,0.2]
        
        qx = torch.sqrt(torch.clamp(x, eps, 1))
        qy = torch.sqrt(torch.clamp(y, eps, 1))
        
        xx = torch.matmul(qx, qx.t())
        yy = torch.matmul(qy, qy.t())
        xy = torch.matmul(qx, qy.t())
        
        off_diag = 1 - torch.eye(n).to(device)
        
        def diffusion_kernel(a, tmpt, dim):
            return torch.exp(-torch.acos(torch.clamp(a, 0, 1 - eps)).pow(2)) / tmpt

        sum_xx_total, sum_yy_total, sum_xy_total = 0, 0, 0
        
        for t in t_values:
            k_xx = diffusion_kernel(xx, t, d - 1)
            k_yy = diffusion_kernel(yy, t, d - 1)
            k_xy = diffusion_kernel(xy, t, d - 1)
            
            sum_xx = (k_xx * off_diag).sum() / (n * (n - 1))
            sum_yy = (k_yy * off_diag).sum() / (n * (n - 1))
            sum_xy = 2 * k_xy.sum() / (n * n)
            
            sum_xx_total += sum_xx
            sum_yy_total += sum_yy
            sum_xy_total += sum_xy
        
        mmd_loss = (sum_xx_total + sum_yy_total - sum_xy_total) / len(t_values)

    return mmd_loss


def compute_dirichlet_likelihood(alphas, posterior_theta):
    """
    Reference:
        - Maier, M. (2014). DirichletReg: Dirichlet regression for compositional data in R.
    """
    N = alphas.shape[0]
    n_topics = alphas.shape[1]
    transformed_thetas = 1 / N * (posterior_theta * (N - 1) + 1 / n_topics)
    LL = (
        alphas.sum(1).lgamma()
        - alphas.lgamma().sum(1)
        + ((alphas - 1) * torch.log(transformed_thetas)).sum(1)
    )
    return LL.sum()


def top_k_indices_column(col, k):
    """
    Returns the indices of the top k largest values for each column in a NumPy array.
    """
    return np.argsort(col)[-k:]


def topic_diversity(topic_words, topK=10):
    """
    Computes the topic diversity of a topic-word matrix (measured as the proportion of unique words in the top-k words of each topic).
    """
    unique_words = set()
    for topic in topic_words:
        unique_words = unique_words.union(set(topic[:topK]))
    td = len(unique_words) / (topK * len(topic_words))
    return td


def parse_modality_view(key: str):
    if '_' not in key:
        raise ValueError(f"Expected 'modality.view', got '{key}'")
    return key.split('_', 1)
