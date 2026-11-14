"""Fixed-dictionary recognition preprocessing and greedy decoding."""

import math
from numbers import Integral
from pathlib import Path

import cv2
import numpy as np

from .vocabulary import OFFICIAL_TOKENS


def decode_image(content: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable recognition image")
    return image


def resize_image(image, *, height=48, width=320, inference=False):
    if height < 1 or width < 1 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Expected nonempty BGR image and positive canvas dimensions")
    ratio = image.shape[1] / image.shape[0]
    canvas = int(height * max(width / height, ratio)) if inference else width
    resized_width = min(canvas, math.ceil(height * ratio))
    x = cv2.resize(image, (resized_width, height)).astype(np.float32).transpose(2, 0, 1)
    x /= 255
    x -= 0.5
    x /= 0.5
    output = np.zeros((3, height, canvas), dtype=np.float32)
    output[:, :, :resized_width] = x
    return output, resized_width / canvas


def preprocess(content: bytes, *, inference: bool = False) -> tuple[np.ndarray, float]:
    return resize_image(decode_image(content), inference=inference)


class Vocabulary:
    def __init__(self, dictionary: Path):
        with dictionary.open("rb") as stream:
            characters = [line.decode("utf-8").rstrip("\r\n") for line in stream]
        if len(characters) != 18708:
            raise ValueError("Expected the PP-OCRv6 medium dictionary")
        self.characters = ["blank", *characters, " "]
        self.indices = {value: index for index, value in enumerate(self.characters)}
        self.tokens = OFFICIAL_TOKENS
        self.nrtr_indices = {token: i + 3 for token, i in self.indices.items() if i != self.tokens.blank}

    @classmethod
    def from_symbols(cls, ctc_symbols, nrtr_symbols, tokens):
        """Build a code-point encoding from explicit class-index mappings."""
        out = cls.__new__(cls)
        out.tokens = tokens
        out.indices = dict(ctc_symbols)
        out.nrtr_indices = dict(nrtr_symbols)
        if set(out.indices) != set(out.nrtr_indices):
            raise ValueError("Both heads must encode the same code points")
        for symbols, reserved, size in ((out.indices, {tokens.blank}, tokens.ctc_classes),
                                        (out.nrtr_indices, {tokens.pad, tokens.bos, tokens.eos}, tokens.nrtr_classes)):
            if (len(set(symbols.values())) != len(symbols) or any(len(k) != 1 for k in symbols)
                    or any(not isinstance(i, Integral) or isinstance(i, bool) or i in reserved
                           or not 0 <= i < size for i in symbols.values())):
                raise ValueError("Invalid recognition class mapping")
        if set(out.indices.values()) | {tokens.blank} != set(range(tokens.ctc_classes)):
            raise ValueError("CTC mapping must cover every nonblank class")
        out.characters = [""] * tokens.ctc_classes
        for symbol, index in out.indices.items():
            out.characters[index] = symbol
        return out

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray, int]:
        symbols = [c for c in text if c in self.indices and c in self.nrtr_indices]
        indices = [self.indices[c] for c in symbols]
        if len(text) > 25 or not 1 <= len(indices) <= 23:
            raise ValueError(f"Invalid recognition target of length {len(text)}")
        count = len(indices)
        ctc = np.full(25, self.tokens.blank, np.int64)
        nrtr = np.full(25, self.tokens.pad, np.int64)
        ctc[:count] = indices
        nrtr[:count + 2] = [self.tokens.bos, *(self.nrtr_indices[c] for c in symbols), self.tokens.eos]
        return ctc, nrtr, count

    def decode(self, probabilities: np.ndarray) -> list[tuple[str, float]]:
        if probabilities.ndim != 3 or probabilities.shape[-1] != self.tokens.ctc_classes:
            raise ValueError("Probability shape does not match CTC vocabulary")
        results = []
        for indices, confidence in zip(probabilities.argmax(2), probabilities.max(2)):
            keep = indices != self.tokens.blank
            keep[1:] &= indices[1:] != indices[:-1]
            results.append(("".join(self.characters[i] for i in indices[keep]),
                            float(confidence[keep].mean()) if keep.any() else 0.0))
        return results
