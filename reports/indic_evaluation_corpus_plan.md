# Indic OCR evaluation corpus planning

Keep this design separate from the Phase 3.7 numerical fixture. Start with a fixed-checkpoint recognition baseline on real, licensed word and line images. Confirm each candidate corpus against its publisher's records for annotation provenance, redistribution terms, released scripts and official splits before use.

Inventory examples by script and language, printed and handwritten text, scene versus scanned documents, text length, mixed-script tokens, and punctuation. Keep source documents, pages from the same work, and crops from each image in one partition. Preserve official splits and audit grouped IDs, exact hashes and perceptual near duplicates. If author or document links are unavailable, record the limits of the audit.

Freeze a separate validation manifest for model selection and a held-out test manifest for final evaluation. Record sample ID, license, attribution, source group, image hash, manifest hash and the raw transcription. Publish aggregate counts and reproduction instructions under the applicable terms. Report raw and NFC code-point CER, grapheme-cluster error rate, WER and script-specific coverage. Include labels outside the existing checkpoint vocabulary when measuring baseline coverage. The frozen HierText result provides no evidence of Indic OCR quality.

## Candidate publisher releases

| Dataset | Publisher evidence and grouping checks |
|---|---|
| IndicSTR12 | Over 27,000 natural-scene word images across 12 languages. The paper covers Assamese, Bengali, Odia, Marathi, Hindi, Kannada, Urdu, Telugu, Malayalam, Tamil, Gujarati and Punjabi. Verify archive rights, split files and parent-image IDs before ingestion. Treat overlap with later scene collections as possible until audited. [Paper](https://arxiv.org/abs/2403.08007) |
| WIRIndic 2024 | Ten language sets with 25,000 road-scene word crops each. The publisher lists 17,500 train, 2,500 validation and 5,000 test per language. Public links and train/validation label filenames are documented. The test description lists images and filenames, with no downloadable ground truth confirmed in this audit. Same-scene and street-view sequence grouping and redistribution permission require verification. [Dataset page](https://ilocr.iiit.ac.in/icpr_2024_wirindic/dataset.html) |
| IIIT-INDIC-HW-WORDS | The authors report roughly 868k word instances from 135 writers for eight newly collected scripts, with earlier Devanagari and Telugu collections discussed separately. Confirm whether downloaded splits are writer-disjoint and whether archive terms permit local evaluation and redistribution of derived crops. [Authors' paper](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2021/iiit-indic-hw-words.pdf) |
| ICDAR 2025 IHDR | The organizers describe about 10,000 captured document images and 400,000 words across ten Indic languages. Documents provide a separate handwriting domain and can be grouped by writer and page where IDs are available. Verify release availability, ground truth access and license before using this candidate. [Publisher](https://ilocr.iiit.ac.in/icdar_2025_Indic_HDR/) |

These checks establish published dataset descriptions and split metadata where available. Archive bytes, signed access conditions, and explicit reuse licenses were not verified for the candidates. A visible download link is insufficient evidence for redistribution. Retain training and validation splits separately and obtain writer or parent-image metadata before creating any new partitions.
