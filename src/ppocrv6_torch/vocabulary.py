"""Recognition head dimensions and special token indices."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecognitionTokens:
    ctc_classes: int = 18710
    nrtr_classes: int = 18714
    blank: int = 0
    pad: int = 0
    bos: int = 2
    eos: int = 3

    def __post_init__(self):
        special = (self.pad, self.bos, self.eos)
        if self.ctc_classes < 2 or not 0 <= self.blank < self.ctc_classes:
            raise ValueError("Invalid CTC class count or blank index")
        if (self.nrtr_classes < 4 or len(set(special)) != 3
                or any(not 0 <= index < self.nrtr_classes for index in special)):
            raise ValueError("Invalid NRTR class count or special token indices")


OFFICIAL_TOKENS = RecognitionTokens()
