from typing import Dict, OrderedDict

import numpy as np

from tiledb.vector_search.embeddings import ObjectEmbedding

EMBED_DIM = 512


class SomaScGPTEmbedding(ObjectEmbedding):
    def __init__(
        self,
        model_uri: str,
        gene_col: str = "feature_name",
        device: str = "cpu",
        max_length: int = 1200,
        batch_size: int = 64,
        use_batch_labels: bool = False,
        use_fast_transformer: bool = False,
    ):
        self.model_uri = model_uri
        self.gene_col = gene_col
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_batch_labels = use_batch_labels
        self.use_fast_transformer = use_fast_transformer
        self.model = None
        self.model_configs = None
        self.vocab = None

    def init_kwargs(self) -> Dict:
        return {
            "model_uri": self.model_uri,
            "gene_col": self.gene_col,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "use_batch_labels": self.use_batch_labels,
            "use_fast_transformer": self.use_fast_transformer,
        }

    def dimensions(self) -> int:
        return EMBED_DIM

    def vector_type(self) -> np.dtype:
        return np.float32

    def load(self) -> None:
        import io
        import json

        import torch
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained

        import tiledb

        if self.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available():
                print("WARNING: CUDA is not available. Using CPU instead.")

        obj_type = tiledb.object_type(self.model_uri)
        if obj_type == "group":
            with tiledb.Group(self.model_uri) as grp:
                for child in list(grp):
                    with tiledb.open(child.uri, "r") as a:
                        if a.meta["original_file_name"] == "args.json":
                            model_config_file_uri = child.uri
                        elif a.meta["original_file_name"] == "best_model.pt":
                            best_model_uri = child.uri
                        elif a.meta["original_file_name"] == "vocab.json":
                            vocab_uri = child.uri
            model_config_file = tiledb.filestore.Filestore(model_config_file_uri)
            model_file = tiledb.filestore.Filestore(best_model_uri)
            vocab_file = tiledb.filestore.Filestore(vocab_uri)
        else:
            vfs = tiledb.VFS()
            model_config_file_uri = f"{self.model_uri}/args.json"
            best_model_uri = f"{self.model_uri}/best_model.pt"
            vocab_uri = f"{self.model_uri}/vocab.json"
            model_config_file = vfs.open(model_config_file_uri)
            model_file = vfs.open(best_model_uri)
            vocab_file = vfs.open(vocab_uri)

        # Load model
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        vocab_dict = json.load(vocab_file)
        self.vocab = GeneVocab.from_dict(vocab_dict)
        for s in special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.model_configs = json.load(model_config_file)
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.model_configs["embsize"],
            nhead=self.model_configs["nheads"],
            d_hid=self.model_configs["d_hid"],
            nlayers=self.model_configs["nlayers"],
            nlayers_cls=self.model_configs["n_layers_cls"],
            n_cls=1,
            vocab=self.vocab,
            dropout=self.model_configs["dropout"],
            pad_token=self.model_configs["pad_token"],
            pad_value=self.model_configs["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=self.use_fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=False,
        )
        load_pretrained(
            self.model,
            torch.load(io.BytesIO(model_file.read()), map_location=self.device),
            verbose=False,
        )
        self.model.to(self.device)
        self.model.eval()

    def embed(self, objects: OrderedDict, metadata: OrderedDict) -> np.ndarray:
        import multiprocessing

        import numpy as np
        import torch
        from scgpt.data_collator import DataCollator
        from torch.utils.data import DataLoader
        from torch.utils.data import SequentialSampler
        from tqdm import tqdm

        from tiledb.vector_search.embeddings.soma_scgpt_embedding import (
            ScGPTTorchDataset,
        )

        adata = objects["anndata"]
        adata.var["id_in_vocab"] = [
            self.vocab[gene] if gene in self.vocab else -1
            for gene in adata.var[self.gene_col]
        ]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        if adata.n_obs == 0:
            return np.empty((0, 0))

        genes = adata.var[self.gene_col].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)

        # Embedding using the scGPT utility.
        # This fails on MacOS with pickling errors and
        # AttributeError: module 'os' has no attribute 'sched_getaffinity'

        # from scgpt.tasks import get_batch_cell_embeddings
        # return get_batch_cell_embeddings(
        #     adata,
        #     cell_embedding_mode="cls",
        #     model=self.model,
        #     vocab=self.vocab,
        #     max_length=self.max_length,
        #     batch_size=self.batch_size,
        #     model_configs=self.model_configs,
        #     gene_ids=gene_ids,
        #     use_batch_labels=self.use_batch_labels,
        # )

        # This is mostly a copy of `scgpt.tasks.get_batch_cell_embeddings`.
        # The differences are:
        # - Using `multiprocessing.cpu_count` instead of `os.sched_getaffinity`
        # - Moving the local `Dataset` class to `ScGPTTorchDataset` to avoid
        #   pickling error on MacOs.
        count_matrix = adata.X
        count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        if self.use_batch_labels:
            batch_ids = np.array(adata.obs["batch_id"].tolist())

        dataset = ScGPTTorchDataset(
            count_matrix,
            gene_ids,
            self.vocab,
            self.model_configs,
            batch_ids if self.use_batch_labels else None,
        )
        collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.model_configs["pad_token"]],
            pad_value=self.model_configs["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=self.max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=min(multiprocessing.cpu_count(), self.batch_size),
            pin_memory=True,
        )

        cell_embeddings = np.zeros(
            (len(dataset), self.model_configs["embsize"]), dtype=np.float32
        )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            for data_dict in tqdm(data_loader, desc="Embedding cells"):
                input_gene_ids = data_dict["gene"].to(self.device)
                src_key_padding_mask = input_gene_ids.eq(
                    self.vocab[self.model_configs["pad_token"]]
                )
                embeddings = self.model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(self.device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=data_dict["batch_labels"].to(self.device)
                    if self.use_batch_labels
                    else None,
                )

                embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count : count + len(embeddings)] = embeddings
                count += len(embeddings)
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
        return cell_embeddings


# Copy of `scgpt.tasks.get_batch_cell_embeddings.Dataset` class.
class ScGPTTorchDataset:
    def __init__(self, count_matrix, gene_ids, vocab, model_configs, batch_ids=None):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.vocab = vocab
        self.model_configs = model_configs
        self.batch_ids = batch_ids

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        import torch

        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        # append <cls> token at the beginning
        genes = np.insert(genes, 0, self.vocab["<cls>"])
        values = np.insert(values, 0, self.model_configs["pad_value"])
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output
