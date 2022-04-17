import os
import pickle
from typing import List, Tuple

import faiss
import gdown
import numpy as np

from .search import Base


class SearchSolution(Base):
    def __init__(
        self,
        data_file="./data/train_data.pickle",
        data_url="https://drive.google.com/uc?id=1D_jPx7uIaCJiPb3pkxcrkbeFcEogdg2R",
        nlist=512,  # the number of cells
        nprobe=64,  # the number of cells (out of nlist) that are visited to perform a search
        top_k=8,  # nearest neighbors
        dim=512,
    ) -> None:
        self.data_file = data_file
        self.data_url = data_url
        self.top_k = top_k
        self.nlist = nlist
        self.dim = dim
        self.quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )
        self.index.nprobe = nprobe

    def add_vectors2index(self, vectors: np.array) -> None:
        self.index.add(vectors)

    def set_base_from_pickle(self):
        """
        Downloads the data, if it does not exist.
        Sets reg_matrix and pass_dict

        reg_matrix : np.array(N, 512)
        pass_dict : dict -> dict[idx] = [np.array[1, 512]]
        """
        if not os.path.isfile(self.data_file):
            if not os.path.isdir("./data"):
                os.mkdir("./data")
            gdown.download(self.data_url, self.data_file, quiet=False)

        with open(self.data_file, "rb") as f:
            data = pickle.load(f)

        self.reg_matrix = [None] * len(data["reg"])
        self.ids = {}
        for i, key in enumerate(data["reg"]):
            self.reg_matrix[i] = data["reg"][key][0][None]
            self.ids[i] = key

        self.reg_matrix = np.concatenate(self.reg_matrix, axis=0).astype("float32")
        faiss.normalize_L2(self.reg_matrix)
        self.index.train(self.reg_matrix)
        self.add_vectors2index(self.reg_matrix)
        self.pass_dict = data["pass"]

    def search(self, query: np.array) -> List[Tuple]:
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)
        query = query.astype("float32")
        D, I = self.index.search(query, self.top_k)
        idxs = [self.ids[id] for id in I[0]]
        return list(zip(idxs, D[0]))

    def insert(self, feature: np.array) -> None:
        if feature.ndim == 1:
            feature = np.expand_dims(feature, axis=0)
        feature = feature.astype("float32")
        faiss.normalize_L2(feature)
        self.add_vectors2index(feature)
        self.ids[len(self.ids)] = feature
        self.reg_matrix = np.concatenate((self.reg_matrix, feature), axis=0)

    def cos_sim(self, query: np.array) -> np.array:
        pass
