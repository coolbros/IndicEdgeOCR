# Dataset provenance

## PaddleOCR multilingual dataset listing

On 2026-11-13, the [PaddleOCR datasets page](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/datasets/ocr_datasets.en.md) listed a generic multilingual recognition entry and said annotations are included in its downloaded image ZIP. The [Baidu Drive share](https://pan.baidu.com/s/1bS_u207Rm7YbY33wOECKDA) returned an extraction-code prompt, including when requested with the documented code `frgi` in the query. Archive retrieval did not succeed. Contents, languages, scripts, image and label counts, file format, official partitions, document grouping, usage and redistribution terms remain unverified.

```bash
curl -fL 'https://pan.baidu.com/s/1bS_u207Rm7YbY33wOECKDA?pwd=frgi' -o artifacts/baidu-share.html
```

This command reproduces the share-page check. The PP-OCRv6 paper describes a curated large-scale training dataset and additional collection for industrial scenarios, and its language appendix describes model coverage. Its benchmark uses an in-house evaluation set. The paper, appendices, model YAML and release documentation reviewed here did not identify a complete downloadable PP-OCRv6 training corpus or connect the generic multilingual listing to that corpus. [Paper §3.1 and Appendix A](https://arxiv.org/html/2606.13108v1), [model release documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md).

## HierText fallback

HierText is an independent public OCR dataset by Long et al. derived from Open Images. It provides word, line and paragraph annotations for scene text and documents, with handwritten and printed text flags. The release does not provide per-word language or script identifiers. The smoke subset is selected for legible horizontal words representable by the fixed PP-OCRv6 dictionary. It is a convenience subset for engineering tests, with no claim about corpus representativeness or whether an upstream pretrained model encountered these images.

The pinned annotation release at `70b6620b2b112597d8219e11eee9773a1403827c` contains 8,281 train, 1,724 validation and 1,634 test source images. The creators report about 1.2 million word annotations. The JPG basenames and annotation `image_id` fields identify original Open Images assets, and annotations store polygon vertices, transcription and legibility. Grouping is available by source image. Higher-level grouping across separate photos or scans of the same physical document is unavailable. [HierText release](https://github.com/google-research-datasets/hiertext/tree/70b6620b2b112597d8219e11eee9773a1403827c).

The publisher releases HierText under CC BY-SA 4.0. Shared adaptations require attribution, an indication of changes, and ShareAlike terms. The underlying Open Images photos have source attribution and licensing considerations. The job retrieves the official data directly into its private working environment. The repository publishes code, checksums and aggregate audit results. [HierText license](https://github.com/google-research-datasets/hiertext/blob/70b6620b2b112597d8219e11eee9773a1403827c/LICENSE), [Open Images source](https://storage.googleapis.com/openimages/web/index.html).

### Retrieval and preparation

Downloaded on 2026-11-13. The script indexes official partitions before cropping. It reads the first 8 MiB of each official image archive and accepts the first four available source images with an eligible crop per partition. It takes the first eligible axis-aligned polygon crop and preserves labels exactly. Filters exclude illegible or vertical words, unsupported characters, words outside 3–18 characters, crops smaller than 14 pixels on either axis, and taller-than-wide crops.

```bash
mkdir -p artifacts/hiertext/raw
pin=70b6620b2b112597d8219e11eee9773a1403827c
for split in train validation test; do
  curl -fL "https://raw.githubusercontent.com/google-research-datasets/hiertext/${pin}/gt/${split}.jsonl.gz" \
    -o "artifacts/hiertext/raw/${split}.jsonl.gz"
  curl -fL --range 0-8388607 "https://open-images-dataset.s3.amazonaws.com/ocr/${split}.tgz" \
    -o "artifacts/hiertext/raw/${split}.tgz.prefix"
done
PYTHONPATH=src python tools/prepare_hiertext_subset.py \
  --raw artifacts/hiertext/raw --output artifacts/hiertext/subset \
  --dictionary references/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt \
  --retrieved-on 2026-11-13 --per-split 4
```
