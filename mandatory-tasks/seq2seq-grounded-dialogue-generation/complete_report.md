# From Translation to Grounded Dialogue Generation

All models were built from scratch in PyTorch (no pretrained transformers, no ready-made seq2seq pipelines) and trained on a CPU, so they are small on purpose. All scores are from a single run.

## Subtask 1: Basic Seq2Seq Translation (English to French)

- **Data:** 20,000 training pairs, 275 validation pairs, 1,000 test pairs. Sentences up to 15 tokens. Word-level vocabulary of 5,000 words on each side.
- **Model:** LSTM encoder compresses the sentence into one context vector. LSTM decoder generates the French sentence from it. 3.36M parameters, 8 epochs, greedy decoding.
- **Result:** Test BLEU = **7.76**.
- **Where it struggles:**
  - Long sentences: BLEU drops from 10.71 (1-5 words) to 9.23 (6-10 words) to 5.99 (11-15 words). One fixed vector cannot hold a long sentence.
  - Rare words: 16.4% of generated words are `<unk>`, while only 8.3% of target words are out of vocabulary.
  - Repetition in 12.4% of outputs, and outputs are slightly shorter than references (8.9 vs 9.9 words).

## Subtask 2: Attention and Decoding Strategies

- **Change:** Same data and settings, but the decoder now uses Luong attention over all encoder states. 3.55M parameters, 8 epochs.
- **Results (test BLEU):**
  - No attention, greedy: 7.76
  - Attention, greedy: **13.62**
  - Attention, beam search (3 beams): **16.52** (best)
  - Attention, beam search (5 beams): 16.20
  - Attention, top-k sampling: 9.24
- **Attention helps most on long sentences:** BLEU for 11-15 words went from 5.99 to 11.56.
- **Decoding trade-off:** Beam search is best but slow (about 70-80 seconds vs 3 seconds for greedy). Sampling is more varied but scores lower.
- **Not fixed by attention:** `<unk>` only fell from 16.4% to 14.7%, and repetition went up slightly (12.4% to 14.2%). These come from the small word-level vocabulary, so subword units would be the next step.

## Subtask 3: Document-Grounded Dialogue in Hinglish

- **Data:** CMU Hinglish DoG. The Hugging Face version has no document text and no conversation ids, so I downloaded the 30 Wikipedia documents from the original repository and rebuilt the conversations myself. This gave 5,376 training, 645 validation and 686 test examples (previous 3 turns as input, next turn as target).
- **Preprocessing:** Lowercasing, shortening stretched words, merging spelling variants (for example "hein" and "hain" to "hai"), and tagging words as English or Hindi to measure code-mixing. For each example only the 3 most relevant document sentences were kept.
- **Embeddings:** Trained from scratch (skip-gram with negative sampling) on the dialogue and document text, with one shared vocabulary of 10,961 words.
- **Model:** Two BiLSTM encoders (conversation and document) and one decoder that attends over both together with a single softmax. 2.57M parameters.
- **Baselines:** The same model without the document encoder (ungrounded), and a TF-IDF retrieval reply picker (with and without document re-ranking).

**Results (test set, 686 examples):**
- Grounded model (greedy): BLEU 1.14, ROUGE-1 13.22, ROUGE-L 11.40.
- Ungrounded model (greedy): BLEU 1.18, ROUGE-1 12.61, ROUGE-L 11.09.
- Retrieval baselines: BLEU about 0.3, ROUGE-L 8.5 to 9.1.
- BLEU is very low for everyone, which is normal for open-ended dialogue. Replies are generic and repetitive (distinct-2 is 0.02 vs 0.69 for human replies), for example "mujhe lagta hai ki ye movie movie hai".

**Does the model use the document?**
- Yes, but the gain is small. Document overlap (share of generated content words found in the document) was 8.1% with the correct document, 3.6% with a wrong one, 0.3% with an empty one, and 2.6% for the ungrounded model. Swapping the document changed 60% of the replies.
- However, the score difference between grounded and ungrounded is within noise. Attention on the document was also the same for correct and wrong documents (0.577 vs 0.576), so attention weights alone are not good evidence of grounding here.

**Code-mixing:**
- Mixed replies were harder than mostly-Hindi replies (grounded ROUGE-L 10.6 vs 12.9, BLEU 0.63 vs 1.60).
- Only 8 replies were mostly-English, so that group could not be compared fairly.

## Limitations and Takeaways

- Models are small, trained for minutes, and the dialogue data has only about 5,000 training examples.
- The model cannot copy names and facts directly from the document, which limits grounding. A copy mechanism would be the biggest improvement.
- Document sentences were selected using the English version of the conversation, which a real system might not have.
- **Main lessons:** A single context vector is a bottleneck, and attention fixes much of it (BLEU 7.76 to 13.62, and 16.52 with beam search). The same idea extends to two inputs by attending over both together. For dialogue, BLEU alone is not enough, so checks such as swapping the document and measuring document overlap were more useful.
