"""Fixed-dictionary recognition preprocessing and greedy decoding."""

import math
from pathlib import Path

import cv2
import numpy as np


def preprocess(content: bytes, *, inference: bool = False) -> tuple[np.ndarray, float]:
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable recognition image")
    height, width = image.shape[:2]
    ratio = width / height
    canvas = int(48 * max(320 / 48, ratio)) if inference else 320
    resized_width = min(canvas, math.ceil(48 * ratio))
    x = cv2.resize(image, (resized_width, 48)).astype(np.float32).transpose(2, 0, 1)
    x /= 255
    x -= 0.5
    x /= 0.5
    output = np.zeros((3, 48, canvas), dtype=np.float32)
    output[:, :, :resized_width] = x
    return output, resized_width / canvas


class Vocabulary:
    def __init__(self, dictionary: Path):
        with dictionary.open("rb") as stream:
            characters = [line.decode("utf-8").rstrip("\r\n") for line in stream]
        if len(characters) != 18708:
            raise ValueError("Expected the PP-OCRv6 medium dictionary")
        self.characters = ["blank", *characters, " "]
        self.indices = {value: index for index, value in enumerate(self.characters)}

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray, int]:
        indices = [self.indices[c] for c in text if c in self.indices]
        if len(text) > 25 or not 1 <= len(indices) <= 23:
            raise ValueError(f"Invalid recognition target of length {len(text)}")
        count = len(indices)
        ctc = np.zeros(25, np.int64)
        nrtr = np.zeros(25, np.int64)
        ctc[:count] = indices
        nrtr[:count + 2] = [2, *(i + 3 for i in indices), 3]
        return ctc, nrtr, count

    def decode(self, probabilities: np.ndarray) -> list[tuple[str, float]]:
        results = []
        for indices, confidence in zip(probabilities.argmax(2), probabilities.max(2)):
            keep = indices != 0
            keep[1:] &= indices[1:] != indices[:-1]
            results.append(("".join(self.characters[i] for i in indices[keep]),
                            float(confidence[keep].mean()) if keep.any() else 0.0))
        return results
