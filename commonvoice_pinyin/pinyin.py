from typing import Tuple, List

import numpy as np

from pypinyin import pinyin, Style

import unicodedata
import string


class PinyinInput:
    """Pinyin input network"""

    def __call__(self, sentences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a batch from a list of sentences
        Args:
            sentences: a list of sentences
        Returns:
            phoneme batch: [B, L, P]
            lengths: [B]
        """
        assert type(sentences) == list
        batch = len(sentences)
        lengths = []
        phonemes = []
        for b, s in enumerate(sentences):
            s = unicodedata.normalize("NFKC", s)
            # Convert using pypinyin
            pinyins = pinyin(s, Style.TONE3)
            pys = []
            for p in pinyins:
                py = Pinyin(p[0]).to_array()
                pys.append(np.array(py))
            phonemes.append(np.stack(pys))
            lengths.append(len(pys))
        lengths_a = np.array(lengths)
        phonemes_a = self.pad_sequence(phonemes)
        return phonemes_a, lengths_a

    @staticmethod
    def pad_sequence(seqs):
        """
        Pad a list of sequences to longest sequence and return as an ndarray
        Args:
            seqs: List[ndarray], [L, *], highest dimension is sequence length
        """
        max_len = max(map(len, seqs))
        seqs_padded = []
        for s in seqs:
            dims = len(s.shape)
            padding = max_len - len(s)
            padded = np.pad(s, [(0, padding)] + [(0, 0)] * (dims - 1))
            seqs_padded.append(padded)
        return np.stack(seqs_padded)


class Pinyin:
    PUNCTS = (
        string.punctuation +
        '「」、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜–—‘’‛“”„‟・‧、·×。。' +
        " "
    )
    CONSONANTS = "bpmfdtnlgkhjqxzcsryw"
    VOWELS_1 = ["a", "o", "e", "i", "u:", "u"]
    VOWELS_2 = ["a", "o", "e", "r"]
    VOWELS_3 = ["i", "o", "u"]
    NASALS = ["ng", "n"]
    TONES = "12345"
    ALPHABETS = string.ascii_lowercase
    dim = (
        len(PUNCTS) + 1 +  # Optional punctuation
        len(CONSONANTS) + 1 +  # Optional consonant
        2 +  # Retroflex
        len(VOWELS_1) + 1 +  # Optional vowel 1
        2 +  # Er
        len(VOWELS_2) + 1 +  # Optional vowel 2
        len(VOWELS_3) + 1 +  # Optional vowel 3
        len(NASALS) + 1 +  # Optional nasal
        len(TONES) + 1 +  # Optional tone
        len(ALPHABETS) + 1  # Optional alphabet
    )
    shape = (dim,)

    def __init__(self, pinyin: str, validate=True):
        assert unicodedata.is_normalized("NFKC", pinyin)
        self.punct = None
        self.consonant = None
        self.retroflex = False
        self.vowel1 = None
        self.er = False
        self.vowel2 = None
        self.vowel3 = None
        self.nasal = None
        self.tone = None
        self.alphabet = None

        # Punctuation
        self.punct, r = self.match_one_of(self.PUNCTS, pinyin)
        if self.punct is not None:
            return

        # Consonant or semivowel
        self.consonant, r = self.match_consonant(r)
        if self.consonant is not None and self.consonant in "zcs":
            self.retroflex, r = self.match_retroflex(r)

        # Vowel 1
        if self.vowel1 is None:
            self.vowel1, r = self.match_one_of(self.VOWELS_1, r)
        # Er
        if self.vowel1 == "e":
            self.er, r = self.match_er(r)
        self.vowel2, r = self.match_one_of(self.VOWELS_2, r)
        self.vowel3, r = self.match_one_of(self.VOWELS_3, r)
        self.nasal, r = self.match_one_of(self.NASALS, r)
        self.tone, r = self.match_one_of(self.TONES, r)
        is_pinyin, reason = self.is_pinyin()
        # If all fails, try matching single alphabet
        if not is_pinyin:
            self.alphabet, r = self.match_one_of(
                self.ALPHABETS, pinyin.lower())
        if validate and not is_pinyin and self.alphabet is None:
            raise ValueError(pinyin + ": " + reason)

    def is_pinyin(self) -> Tuple[bool, str]:
        if self.punct is not None:
            if self.consonant is not None or \
                    self.retroflex or \
                    self.vowel1 is not None or \
                    self.er or \
                    self.vowel2 is not None or \
                    self.vowel3 is not None or \
                    self.nasal is not None or \
                    self.tone is not None:
                return False, "Punctuation with syllable"
            return True, ""
        if self.retroflex and self.consonant not in "zcs":
            return False, f"Invalid retroflex: {self.consonant + 'h'}"
        if self.vowel1 is None:
            return False, "Vowel 1 absent"
        if self.tone is None:
            return False, "Tone absent"
        if self.er and self.vowel1 != "e":
            return False, f"Invalid -er: {self.vowel1 + 'r'}"
        if self.er and (self.vowel2 is not None or self.vowel3 is not None or self.nasal):
            return False, f"Syllable trailing -er"
        return True, ""

    def to_array(self) -> np.array:
        p = one_hot_optional(self.PUNCTS, self.punct)
        c = one_hot_optional(self.CONSONANTS, self.consonant)
        r = one_hot_bool(self.retroflex)
        v1 = one_hot_optional(self.VOWELS_1, self.vowel1)
        er = one_hot_bool(self.er)
        v2 = one_hot_optional(self.VOWELS_2, self.vowel2)
        v3 = one_hot_optional(self.VOWELS_3, self.vowel3)
        n = one_hot_optional(self.NASALS, self.nasal)
        t = one_hot_optional(self.TONES, self.tone)
        a = one_hot_optional(self.ALPHABETS, self.alphabet)
        arr = np.concatenate((p, c, r, v1, er, v2, v3, n, t, a))
        assert arr.shape == self.shape, f"{arr.shape} != {self.shape}"
        return arr

    @staticmethod
    def match_consonant(p: str):
        if len(p) == 0:
            return None, p
        if p[0] in Pinyin.CONSONANTS:
            return p[0], p[1:]
        return None, p

    @staticmethod
    def match_retroflex(p: str):
        if len(p) == 0:
            return False, p
        if p[0] == "h":
            return True, p[1:]
        return False, p

    @staticmethod
    def match_one_of(vs, p: str):
        for v in vs:
            if p.startswith(v):
                return v, p[len(v):]
        return None, p

    @staticmethod
    def match_er(p: str):
        if len(p) == 0:
            return False, p
        if p[0] == "r":
            return True, p[1:]
        return False, p

    @staticmethod
    def match_tone(p: str):
        if len(p) == 0:
            return None, p
        if p[0] in Pinyin.TONES:
            return p[0], p[1:]
        return None, p


def one_hot_bool(b):
    return np.array([False, True]) if b else np.array([True, False])


def one_hot_optional(domain, element):
    """One-hot with optional"""
    assert element is None or element in domain
    arr = np.zeros(len(domain) + 1, dtype=bool)
    if element is None:
        arr[len(domain)] = True
    else:
        idx = domain.index(element)
        arr[idx] = True
    return arr


if __name__ == "__main__":
    p = Pinyin("zhuang1")
    a = p.to_array()
    print(a.shape)
