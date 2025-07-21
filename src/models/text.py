import os
import orjson
import json
import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod

#from src.model import TextToEmb
import torch.nn as nn
from typing import Dict, List
import torch.nn.functional as F

class TextToEmb(nn.Module):
    def __init__(
        self, modelpath: str, mean_pooling: bool = False, device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = device
        from transformers import AutoTokenizer, AutoModel
        from transformers import logging

        logging.set_verbosity_error()

        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)
        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size

        if mean_pooling:
            self.forward = self.forward_pooling

        # put it in eval mode by default
        self.eval()

        # Freeze the weights just in case
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)

    def train(self, mode: bool = True) -> nn.Module:
        # override it to be always false
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device=None) -> Dict:
        device = device if device is not None else self.device

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(device))
        length = encoded_inputs.attention_mask.to(dtype=bool).sum(1)

        if squeeze:
            x_dict = {"x": output.last_hidden_state[0], "length": length[0]}
        else:
            x_dict = {"x": output.last_hidden_state, "length": length}
        return x_dict

    @torch.no_grad()
    def forward_pooling(self, texts: List[str], device=None) -> Tensor:
        device = device if device is not None else self.device

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        # From: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(device))
        attention_mask = encoded_inputs["attention_mask"]

        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = output["last_hidden_state"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        if squeeze:
            sentence_embeddings = sentence_embeddings[0]
        return sentence_embeddings


class TextEmbeddings(ABC):
    name = ...

    def __init__(
        self,
        modelname: str,
        path: str = "",
        device: str = "cpu",
        preload: bool = True,
        disable: bool = False,
    ):
        self.modelname = modelname
        self.embeddings_folder = os.path.join(path, self.name)
        self.cache = {}
        self.device = device
        self.disable = disable

        if preload and not disable:
            self.load_embeddings()
        else:
            self.embeddings_index = {}

    @abstractmethod
    def load_model(self) -> None:
        ...

    @abstractmethod
    def load_embeddings(self) -> None:
        ...

    @abstractmethod
    def get_embedding(self, text: str) -> Tensor:
        ...

    def __contains__(self, text):
        return text in self.embeddings_index

    def get_model(self):
        model = getattr(self, "model", None)
        if model is None:
            model = self.load_model()
        return model

    def __call__(self, texts):
        if self.disable:
            return texts

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        x_dict_lst = []
        # one at a time here
        for text in texts:
            # Precomputed in advance
            if text in self:
                x_dict = self.get_embedding(text)
            # Already computed during the session
            elif text in self.cache:
                x_dict = self.cache[text]
            # Load the text model (if not already loaded) + compute on the fly
            else:
                model = self.get_model()
                x_dict = model(text)
                self.cache[text] = x_dict
            x_dict_lst.append(x_dict)

        if squeeze:
            return x_dict_lst[0]
        return x_dict_lst


class TokenEmbeddings(TextEmbeddings):
    name = "token_embeddings"

    def load_model(self):
        self.model = TextToEmb(self.modelname, mean_pooling=False, device=self.device)
        return self.model

    def load_embeddings(self):
        self.embeddings_big = torch.from_numpy(
            np.load(os.path.join(self.embeddings_folder, self.modelname + ".npy"))
        ).to(dtype=torch.float, device=self.device)
        self.embeddings_slice = np.load(
            os.path.join(self.embeddings_folder, self.modelname + "_slice.npy")
        )
        self.embeddings_index = load_json(
            os.path.join(self.embeddings_folder, self.modelname + "_index.json")
        )
        self.text_dim = self.embeddings_big.shape[-1]

    def get_embedding(self, text):
        # Precomputed in advance
        index = self.embeddings_index[text]
        begin, end = self.embeddings_slice[index]
        embedding = self.embeddings_big[begin:end]
        x_dict = {"x": embedding, "length": len(embedding)}
        return x_dict


class SentenceEmbeddings(TextEmbeddings):
    name = "sent_embeddings"

    def load_model(self):
        self.model = TextToEmb(self.modelname, mean_pooling=True, device=self.device)
        return self.model

    def load_embeddings(self):
        self.embeddings = torch.from_numpy(
            np.load(os.path.join(self.embeddings_folder, self.modelname + ".npy"))
        ).to(dtype=torch.float, device=self.device)
        self.embeddings_index = load_json(
            os.path.join(self.embeddings_folder, self.modelname + "_index.json")
        )
        assert len(self.embeddings_index) == len(self.embeddings)

        self.text_dim = self.embeddings.shape[-1]

    def get_embedding(self, text):
        index = self.embeddings_index[text]
        embedding = self.embeddings[index]
        return embedding.to(self.device)


def load_json(json_path):
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    return load_json(json_path)


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))


def save_token_embeddings(
    path, modelname="sentence-transformers/all-mpnet-base-v2", device="cuda"
):
    model = TextToEmb(modelname, device=device)
    annotations = load_annotations(path)

    path = os.path.join(path, TokenEmbeddings.name)
    ptpath = os.path.join(path, f"{modelname}.npy")
    slicepath = os.path.join(path, f"{modelname}_slice.npy")
    jsonpath = os.path.join(path, f"{modelname}_index.json")

    # modelname can have folders
    path = os.path.split(ptpath)[0]
    os.makedirs(path, exist_ok=True)

    # fetch all the texts
    all_texts = []
    for dico in annotations.values():
        for lst in dico["annotations"]:
            all_texts.append(lst["text"])

    # remove duplicates
    all_texts = list(set(all_texts))

    # batch of N/10
    nb_tokens = []
    all_texts_batched = np.array_split(all_texts, 100)

    nb_tokens_so_far = 0
    big_tensor = []
    index = []
    for all_texts_batch in tqdm(all_texts_batched):
        x_dict = model(list(all_texts_batch))

        tensor = x_dict["x"]
        nb_tokens = x_dict["length"]

        # remove padding
        tensor_no_padding = [x[:n].cpu() for x, n in zip(tensor, nb_tokens)]
        tensor_concat = torch.cat(tensor_no_padding)

        big_tensor.append(tensor_concat)
        # save where it is
        ends = torch.cumsum(nb_tokens, 0)
        begins = torch.cat((0 * ends[[0]], ends[:-1]))

        # offset
        ends += nb_tokens_so_far
        begins += nb_tokens_so_far
        nb_tokens_so_far += len(tensor_concat)

        index.append(torch.stack((begins, ends)).T)

    big_tensor = torch.cat(big_tensor).cpu().numpy()
    index = torch.cat(index).cpu().numpy()

    np.save(ptpath, big_tensor)
    np.save(slicepath, index)
    print(f"{ptpath} written")
    print(f"{slicepath} written")

    # correspondance
    dico = {txt: i for i, txt in enumerate(all_texts)}
    write_json(dico, jsonpath)
    print(f"{jsonpath} written")


def save_sent_embeddings(
    path, modelname="sentence-transformers/all-mpnet-base-v2", device="cuda"
):
    model = TextToEmb(modelname, mean_pooling=True, device=device)
    annotations = load_annotations(path)

    path = os.path.join(path, SentenceEmbeddings.name)
    ptpath = os.path.join(path, f"{modelname}.npy")
    jsonpath = os.path.join(path, f"{modelname}_index.json")

    # modelname can have folders
    path = os.path.split(ptpath)[0]
    os.makedirs(path, exist_ok=True)

    # fetch all the texts
    all_texts = []
    for dico in annotations.values():
        for lst in dico["annotations"]:
            all_texts.append(lst["text"])

    # remove duplicates
    all_texts = list(set(all_texts))

    # batch of N/10
    all_texts_batched = np.array_split(all_texts, 100)
    embeddings = []
    for all_texts_batch in tqdm(all_texts_batched):
        embedding = model(list(all_texts_batch)).cpu()
        embeddings.append(embedding)

    embeddings = torch.cat(embeddings).numpy()
    np.save(ptpath, embeddings)
    print(f"{ptpath} written")

    # correspondance
    dico = {txt: i for i, txt in enumerate(all_texts)}
    write_json(dico, jsonpath)
    print(f"{jsonpath} written")
