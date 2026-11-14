# Phase 4 Indic readiness

Start with the frozen baseline on a stratified scene-text and handwriting evaluation set. Prioritize Devanagari, Bengali, Tamil, Telugu and Malayalam for initial coverage. Expand to Gujarati, Gurmukhi, Kannada, Odia and Urdu when verified source partitions support them. Track language separately from script, including Assamese and Bengali, Hindi and Marathi, and Urdu's typography and logical text order. The [corpus plan](indic_evaluation_corpus_plan.md) records the candidate domains and access conditions.

Keep the original transcription as recorded data, plus a versioned NFC view. Audit combining-order changes, canonical decompositions, matras, nukta, viramas, ZWJ/ZWNJ, conjuncts, numerals and punctuation before deciding whether any corpus-specific mapping is appropriate. Do not strip marks, rewrite visual-order glyph sequences or fold digit sets without explicit annotation evidence. Count and report malformed clusters and normalization collisions. Pin the segmentation implementation and Unicode version. [Unicode normalization](https://www.unicode.org/reports/tr15/) · [Unicode grapheme segmentation](https://www.unicode.org/reports/tr29/).

Compare three candidates offline before training adapted heads.

1. Unicode code points have minimal vocabulary growth with possible long targets and repeated-unit penalties for CTC.
2. Extended grapheme clusters preserve common user-perceived characters, with a larger long tail and missing-cluster behavior to quantify.
3. A script-aware orthographic unit vocabulary can encode conjuncts and matras as units with a reversible code-point fallback. This requires validating linguistic boundaries and coverage per script.

Build candidate vocabularies from training labels only. Measure token length, OOV rate, fallback rate and CTC minimum alignment length per crop, including one extra time index per neighboring repeated token. Report truncation against NRTR target capacity and the evaluator's handling of unsupported text. Compare code-point, NFC and grapheme CER, WER, exact match, and metrics by script, length, and domain. Keep the original baseline predictions as a paired reference and preserve document, writer and source-image grouping across official train, validation and test.

The Phase 3.8 model allows separate class dimensions and distinct blank, pad, BOS and EOS indices. The Phase 3.8 mapping utility accepts explicit code-point maps for both heads. Grapheme or orthographic segmentation, adapted weight initialization, and fine-tuning belong to Phase 4.
