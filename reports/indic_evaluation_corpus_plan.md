# Indic OCR evaluation corpus planning

Keep this design separate from the Phase 3.7 numerical fixture. Start with a fixed-checkpoint recognition baseline on real, licensed word and line images. Confirm each candidate corpus against its publisher's records for annotation provenance, redistribution terms, released scripts and official splits before use.

Inventory examples by script and language, printed and handwritten text, scene versus scanned documents, text length, mixed-script tokens, and punctuation. Keep source documents, pages from the same work, and crops from each image in one partition. Preserve official splits and audit grouped IDs, exact hashes and perceptual near duplicates. If author or document links are unavailable, record the limits of the audit.

Freeze a separate validation manifest for model selection and a held-out test manifest for final evaluation. Record sample ID, license, attribution, source group, image hash, manifest hash and the raw transcription. Publish aggregate counts and reproduction instructions under the applicable terms. Report raw and NFC code-point CER, grapheme-cluster error rate, WER and script-specific coverage. Include labels outside the existing checkpoint vocabulary when measuring baseline coverage. The frozen HierText result provides no evidence of Indic OCR quality.
