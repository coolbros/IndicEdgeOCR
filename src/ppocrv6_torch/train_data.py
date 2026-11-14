"""Pinned recognition augmentation and deterministic multi-scale batches."""

import random

import cv2
import numpy as np

from .tia import tia_distort, tia_perspective, tia_stretch


def concatenate(image, label, external, *, probability=0.5, max_length=25):
    if random.random() > probability:
        return image, label
    for other, text in external:
        ratio = image.shape[1] / image.shape[0] + other.shape[1] / other.shape[0]
        if len(label) + len(text) > max_length or ratio > 320 / 48:
            break
        def resize(value):
            h, w = value.shape[:2]
            return cv2.resize(value, (max(1, round(w / h * 48)), 48))
        image = np.concatenate((resize(image), resize(other)), axis=1)
        label += text
    return image, label


def augment(image, *, tia=0.4, crop=0.4, blur=0.4, hsv=0.4, jitter=0.4, noise=0.4, reverse=0.4):
    if random.random() <= tia and min(image.shape[:2]) >= 20:
        image = tia_distort(image, random.randint(3, 6))
        image = tia_stretch(image, random.randint(3, 6))
        image = tia_perspective(image)
    if random.random() <= crop and min(image.shape[:2]) >= 20:
        offset = random.randint(1, 8)
        image = image[offset:] if random.randint(0, 1) else image[:-offset]
    if random.random() <= blur:
        kernel = cv2.getGaussianKernel(5, 1, cv2.CV_32F)
        image = cv2.sepFilter2D(image, -1, kernel, kernel)
    if random.random() <= hsv:
        colors = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        delta = 0.001 * random.random() * (1 if random.random() > 0.5000001 else -1)
        colors[:, :, 2] = colors[:, :, 2] * (1 + delta)
        image = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)
    if random.random() <= jitter and min(image.shape[:2]) > 10:
        height, width = image.shape[:2]
        count = int(random.random() * min(height, width) * 0.01)
        source = image
        image = source.copy()
        for i in range(count):
            image[i:, i:] = source[:height - i, :width - i]
    if random.random() <= noise:
        gaussian = np.random.normal(0, 0.1**0.5, image.shape)
        image = np.clip(image + 0.5 * gaussian, 0, 255).astype(np.uint8)
    if random.random() <= reverse:
        image = 255 - image
    return image


def scale_pairs(first_bs=64, scales=((320, 32), (320, 48), (320, 64)), divided_factor=(8, 16)):
    base_width, base_height = scales[0]
    area = base_width * base_height * first_bs
    fw, fh = divided_factor
    shapes = [(w // fw * fw, h // fh * fh) for w, h in scales]
    if any(w <= 0 or h <= 0 for w, h in shapes) or first_bs < 1:
        raise ValueError("Invalid multi-scale configuration")
    return [(w, h, max(1, area // (w * h))) for w, h in shapes]


def multiscale_batches(count, first_bs=64, seed=2026, epoch=0):
    """Single-worker Paddle-style cyclic scale sizes and padded final batch."""
    if count < 1:
        raise ValueError("Cannot sample an empty training set")
    rng = random.Random(seed + epoch)
    indices = list(range(count))
    rng.shuffle(indices)
    shapes = []
    pairs = scale_pairs(first_bs)
    total = 0
    while total < count:
        for width, height, size in pairs:
            shapes.append((width, height, size))
            total += size
            if total >= count:
                break
    rng.shuffle(shapes)
    start = 0
    batches = []
    for width, height, size in shapes:
        batches.append((width, height, [indices[j % count] for j in range(start, start + size)]))
        start += size
    rng.shuffle(batches)
    return batches
