from __future__ import print_function
from typing import List, Dict

import numpy as np
import os

from .common_dataset import CommonDataset
import gzip
import html
import os
from functools import lru_cache
import ftfy
import regex as re
from tqdm import tqdm
import paddle
from ppcls.data import preprocess
from ppcls.data.preprocess import transform
from ppcls.utils import logger
"""
prompt for CLIP
"""
prompt_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


@lru_cache()
def default_bpe():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(
        range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


"""
CLIP text encoder and decoder for the tokens
"""


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str=default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            '<|startoftext|>': '<|startoftext|>',
            '<|endoftext|>': '<|endoftext|>'
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>', )
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[
                        i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b]
                            for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors="replace").replace('</w>', ' ')
        return text


"""
text_loader 
"""


class SentPreProcessor(object):
    def __init__(self, root: str) -> None:
        self.root = root
        with open(os.path.join(self.root, "labels.txt"), "r") as f:
            data = f.readlines()
        _lines = [l.split() for l in data]
        self.categories = [{
            "id": l[1].strip(),
            "name": l[-1].strip().replace("_", " "),
            "wid": l[0].strip()
        } for l in _lines]
        self.categories.sort(key=lambda x: x["wid"])
        self.drop_keys = [
            'External links', 'References', 'Further reading', 'Bibliography'
        ]
        self._tokenizer = SimpleTokenizer(
            os.path.join(self.root, "bpe_simple_vocab_16e6.txt.gz"))
        self.SEP_TOKENS = [267, 269]  # [',', '.']
        self.wikis = None

    def get_clip_text(self):
        if self.wikis is None:
            self.wikis = [
                self._parse_wiki(id) for id in range(len(self.categories))
            ]
        naive_text = [
            self.gen_naive_desc(id) for id in range(len(self.categories))
        ]
        wiki_text = [self._get_text(wiki) for wiki in self.wikis]
        return [
            naive_text[i] + wiki_text[i] for i in range(len(self.categories))
        ]

    def split_text(self, texts):
        pat = re.compile(
            r'(?<!\w\.\w.)(?<!([A-Z][a-z])|([A-Z])\.)(?<=\.|\?)(?=[\sA-Z])',
            re.X)
        sents = []
        for text in texts:
            split_text = pat.split(text)
            split_text = [
                s.strip() for s in split_text
                if s is not None and s.strip() != ''
            ]
            sents.append(split_text)
        return sents

    def gen_naive_desc(self, id):
        texts = [
            template.format(self.categories[id]['name'] + ' ')
            for template in prompt_templates
        ]
        return '\n'.join(texts)

    def tokenize(self, texts, context_length=75):
        """
        modified from CLIP

        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """

        def _tokenize(texts):
            sot_token = self._tokenizer.encoder["<|startoftext|>"]  # 49406
            eot_token = self._tokenizer.encoder["<|endoftext|>"]  # 49407
            all_tokens = [
                [sot_token] + self._tokenizer.encode(text)[:context_length] +
                [eot_token] for text in texts
            ]
            result = paddle.zeros((len(all_tokens), context_length + 2))
            for i, tokens in enumerate(all_tokens):
                if len(tokens) > context_length + 2:
                    raise RuntimeError(
                        f"Input {texts[i]} is too long for context length {context_length}"
                    )
                result[i, :len(tokens)] = paddle.to_tensor(
                    tokens, dtype=paddle.float32)
            return result

        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts[0], List):
            return [_tokenize(text) for text in tqdm(texts)]
        return _tokenize(texts)

    def _get_text(self, wiki: Dict):
        # use all key part of each wiki text except those in drop_keys
        text = wiki["summary"] + "\n"
        text += "\n".join([
            v for k, v in wiki.items() if k not in ["summary"] + self.drop_keys
        ])
        return text

    def _parse_wiki(self, id) -> Dict:
        try:
            with open(os.path.join(self.root, "wiki", f"desc_{id}.txt")) as rf:
                lines = rf.readlines()
        except UnicodeDecodeError:
            with open(
                    os.path.join(self.root, "wiki", f"desc_{id}.txt"),
                    encoding='gbk') as rf:
                lines = rf.readlines()
        lines = [d.strip() for d in lines if d.strip() != '']
        ret_dict = {}
        key = "summary"
        val = ""
        for line in lines:
            if line[:2] == "==":
                ret_dict[key] = val.strip()
                key = line.strip('= ')
                val = ""
            else:
                val += line + '\n'
        ret_dict[key] = val.strip()
        return ret_dict


"""
dataset
"""


class ImageNetLTDataset(CommonDataset):
    """ImageNetDataset

    Args:
        image_root (str): image root, path to `ILSVRC2012`
        cls_label_path (str): path to annotation file `train_list.txt` or 'val_list.txt`
        transform_ops (list, optional): list of transform op(s). Defaults to None.
        delimiter (str, optional): delimiter. Defaults to None.
        relabel (bool, optional): whether do relabel when original label do not starts from 0 or are discontinuous. Defaults to False.
    """

    def __init__(self,
                 image_root,
                 cls_label_path,
                 context_length=75,
                 is_pretrain=False,
                 transform_ops=None,
                 delimiter=None,
                 relabel=False):

        self.delimiter = delimiter if delimiter is not None else " "
        self.relabel = relabel
        self.is_pretrain = is_pretrain
        self.context_length = context_length
        super(ImageNetLTDataset, self).__init__(image_root, cls_label_path,
                                                transform_ops)

        self.text_tokens = self.get_sentence_tokens(self.context_length)
        self.end_idxs = [len(sents) for sents in self.text_tokens]

    def get_sentence_tokens(self, context_length):
        print('using clip text tokens splitted by sentence')
        cache_root = 'cached'
        cache_path = os.path.join(cache_root, 'IMNET_LT_desc_text_sent.pkl')
        clip_token_path = os.path.join(cache_root, 'IMNET_LT_text_tokens.pkl')
        if os.path.exists(clip_token_path):
            text_tokens = paddle.load(clip_token_path)
            return text_tokens

        preprocessor = SentPreProcessor(root=self._img_root)
        if not os.path.exists(cache_path):
            os.makedirs(cache_root, exist_ok=True)
            texts = preprocessor.get_clip_text()
            texts = preprocessor.split_text(texts)
            paddle.save(texts, cache_path)
        else:
            texts = paddle.load(cache_path)
        text_tokens = preprocessor.tokenize(
            texts, context_length=context_length)
        paddle.save(text_tokens, clip_token_path)
        return text_tokens

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            if self.is_pretrain:
                return ((img, self.labels[idx]), self.labels[idx])
            return (img, self.labels[idx])

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def _load_anno(self, seed=None):
        assert os.path.exists(
            self._cls_path), f"path {self._cls_path} does not exist."
        assert os.path.exists(
            self._img_root), f"path {self._img_root} does not exist."
        assert os.path.exists(os.path.join(self._img_root,
                                           "wiki")), f"wiki does not exist."
        assert os.path.exists(os.path.join(
            self._img_root, "labels.txt")), f"labels does not exist."
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if self.relabel:
                label_set = set()
                for line in lines:
                    line = line.strip().split(self.delimiter)
                    label_set.add(np.int64(line[1]))
                label_map = {
                    oldlabel: newlabel
                    for newlabel, oldlabel in enumerate(label_set)
                }

            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for line in lines:
                line = line.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, line[0]))
                if self.relabel:
                    self.labels.append(label_map[np.int64(line[1])])
                else:
                    self.labels.append(np.int64(line[1]))
                if os.path.exists(self.images[-1]) == False:
                    self.images.pop()
