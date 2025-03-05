#!/usr/bin/env python
import torch
torch.backends.cudnn.deterministic = True
import os
import yaml
import pickle
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Union, Optional, Dict

from back_translation.bt_model import build_model, SignModel


SIL_TOKEN = "<si>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


def load_pickle_file(data_path: Union[str, Path]) -> dict:
    with open(data_path, "rb") as handle:
        csv_data = pickle.load(handle)
    return csv_data


def load_config(path: Union[Path, str] = "configs/default.yaml") -> Dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    if isinstance(path, str):
        path = Path(path)
    with path.open("r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def get_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Returns the latest checkpoint (by creation time, not the steps number!)
    from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = ckpt_dir.glob("*.ckpt")
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=lambda f: f.stat().st_ctime)

    # check existence
    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in directory {ckpt_dir}.")
    return latest_checkpoint


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self):
        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size
        self.specials = []
        self.itos = []
        self.stoi = None
        self.DEFAULT_UNK_ID = None

    def _from_list(self, tokens: List[str] = None):
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.

        :param tokens: list of tokens
        """
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file: str):
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.

        :param file: path to file where the vocabulary is loaded from
        """
        tokens = []
        with open(file, "r", encoding="utf-8") as open_file:
            for line in open_file:
                tokens.append(line.strip("\n"))
        self._from_list(tokens)

    def __str__(self) -> str:
        return self.stoi.__str__()

    def to_file(self, file: str):
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        :param file: path to file where the vocabulary is written
        """
        with open(file, "w", encoding="utf-8") as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))

    def add_tokens(self, tokens: List[str]):
        """
        Add list of tokens to vocabulary

        :param tokens: list of tokens to add to the vocabulary
        """
        for t in tokens:
            new_index = len(self.itos)
            # add to vocab if not already there
            if t not in self.itos:
                self.itos.append(t)
                self.stoi[t] = new_index

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary

        :param token:
        :return: True if covered, False otherwise
        """
        return self.stoi[token] == self.DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)


class GlossVocabulary(Vocabulary):
    def __init__(self, tokens: List[str] = None, file: str = None):
        """
        Create vocabulary from list of tokens or file.

        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        super().__init__()
        self.specials = [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN]
        self.DEFAULT_UNK_ID = lambda: 1
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)

        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

        # TODO (Cihan): This bit is hardcoded so that the silence token
        #   is the first label to be able to do CTC calculations (decoding etc.)
        #   Might fix in the future.
        assert self.stoi[SIL_TOKEN] == 0

    def arrays_to_sentences(self, arrays: np.array) -> List[List[str]]:
        gloss_sequences = []
        for array in arrays:
            sequence = []
            for i in array:
                sequence.append(self.itos[i])
            gloss_sequences.append(sequence)
        return gloss_sequences


class TextVocabulary(Vocabulary):
    def __init__(self, tokens: List[str] = None, file: str = None):
        """
        Create vocabulary from list of tokens or file.

        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        super().__init__()
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.DEFAULT_UNK_ID = lambda: 0
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)

        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def array_to_sentence(self, array: np.array, cut_at_eos=True) -> List[str]:
        """
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(self.array_to_sentence(array=array, cut_at_eos=cut_at_eos))
        return sentences


def build_vocab(
    field: str, vocab_file: str = None
) -> Vocabulary:
    """
    Builds vocabulary for a torchtext `field` from given`dataset` or
    `vocab_file`.

    :param field: attribute e.g. "src"
    :param vocab_file: file to store the vocabulary,
        if not None, load vocabulary from here
    :return: Vocabulary created from either `dataset` or `vocab_file`
    """

    if vocab_file is not None:
        # load it from file
        if field == "gls":
            vocab = GlossVocabulary(file=vocab_file)
        elif field == "txt":
            vocab = TextVocabulary(file=vocab_file)
        else:
            raise ValueError("Unknown vocabulary type")
    else:
        raise ValueError("Vecabulary file needed")

    for i, s in enumerate(vocab.specials):
        if i != vocab.DEFAULT_UNK_ID():
            assert not vocab.is_unk(s)

    return vocab


class translation_bacth():
    def __init__(self, sgn, sgn_mask, sgn_length):
        self.sgn = sgn
        self.sgn_mask = sgn_mask
        self.sgn_lengths = sgn_length

    def make_cuda(self):
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()
        self.sgn_lengths = self.sgn_lengths.cuda()


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location="cuda" if use_cuda else "cpu", weights_only=False)
    return checkpoint


def make_back_translation_model(model_dir: Union[str, Path]):
    translation_model = Path(model_dir)
    cfg_file = translation_model / "config.yaml"
    ckpt = translation_model / "best.ckpt"
    if not ckpt.exists():
        ckpt = get_latest_checkpoint(translation_model)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(translation_model)
            )
    cfg = load_config(cfg_file)

    use_cuda = cfg["training"].get("use_cuda", False)

    # load the data
    data_cfg = cfg["data"]

    max_output_len = data_cfg.get("max_sent_length", None)
    gls_vocab_file = translation_model / 'gls.vocab'
    txt_vocab_file = translation_model / 'txt.vocab'
    if not gls_vocab_file.exists():
        gls_vocab_file = data_cfg.get("gls_vocab", None)
        if gls_vocab_file is None:
            raise FileNotFoundError(
                "No gloss vocabulary found in directory {}.".format(translation_model)
            )
    if not txt_vocab_file.exists():
        txt_vocab_file = data_cfg.get("txt_vocab", None)
        if txt_vocab_file is None:
            raise FileNotFoundError(
                "No text vocabulary found in directory {}.".format(translation_model)
            )

    gls_vocab = build_vocab(
        field="gls",
        vocab_file=gls_vocab_file,
    )
    txt_vocab = build_vocab(
        field="txt",
        vocab_file=txt_vocab_file,
    )

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    # do_recognition = False
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0

    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )
    model.load_state_dict(model_checkpoint["model_state"])
    model.eval()

    model.max_output_len = max_output_len
    # TODO: chekc this is 30
    if use_cuda:
        model.cuda()

    model.beam_alpha = -1
    model.beam_size = 3

    model.subsample = cfg['data']['skeleton_subsample']

    return model


# pylint: disable-msg=logging-too-many-args
def back_translate(model: SignModel, poses: torch.Tensor
) -> List:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param model: a sign back translation model
    :param poses: skeleton poses
    """
    batch_size = 32
    # load data
    sgn = poses
    max_len = max([len(s) for s in sgn])
    sgn = [torch.flatten(s, start_dim=-2, end_dim=-1) for s in sgn] # flatten
    sgn_length = [s.shape[0] for s in sgn]

    _, features_dim = sgn[0].shape
    sgn = [torch.cat((s, torch.zeros((max_len-s.shape[0], features_dim)))) for s in sgn]  # pad
    sgn = torch.stack(sgn, dim=0)
    sgn_mask = torch.ones(sgn.shape[0], 1, max_len).bool()
    for i, l in enumerate(sgn_length):
        sgn_mask[i, 0, l:] = False

    sgn_length = torch.Tensor(sgn_length).to(torch.float32)

    # NOTE (Cihan): Currently Hardcoded to be 0 for TensorFlow decoding
    assert model.gls_vocab.stoi[SIL_TOKEN] == 0

    model.do_recognition = False
    model.eval()
    model.training = False
    all_txt_outputs = []
    with torch.no_grad():
        # batch sgn
        num_batches = len(sgn) // batch_size
        if len(sgn) % batch_size != 0:
            num_batches += 1
        start = 0
        end = batch_size
        for i in tqdm(range(num_batches), desc='Running Back Translation', total=num_batches):
            _sgn = sgn[start:end]
            _sgn_mask = sgn_mask[start:end]
            _sgn_length = sgn_length[start:end]
            start = end
            end += batch_size

            batch = translation_bacth(_sgn, _sgn_mask, _sgn_length)
            batch.make_cuda()
            (
                batch_gls_predictions,
                batch_txt_predictions,
                batch_attention_scores,
            ) = model.run_batch(
                batch=batch,
                recognition_beam_size=None,
                translation_beam_size=model.beam_size,
                translation_beam_alpha=model.beam_alpha,
                translation_max_output_length=model.max_output_len,
            )
            all_txt_outputs.extend(batch_txt_predictions)
        # decode back to symbols
        decoded_txt = model.txt_vocab.arrays_to_sentences(arrays=all_txt_outputs)
        decoded_txt = [' '.join(t) for t in decoded_txt]
    return decoded_txt
