# refs_new notes (verified 2026-11-12)

How each entry was checked: title and full author list were checked against the arXiv API (export.arxiv.org) and/or Crossref (api.crossref.org).
Venue was confirmed from the arXiv journal_ref/comment, the Crossref DOI, or the venue's own page (NeurIPS/ICLR/PMLR).
Code URLs were checked to resolve on GitHub (HTTP 200) and, where possible, the README was read.
"n/s" = not stated in the paper.

Dropped because they could not be verified or are out of scope:
- "Gram": no EEG foundation model with this name was found.
- BrainBERT (ICLR 2023): intracranial, not scalp EEG.
- TSception (IEEE TAC 2022): before 2023.
- Gui et al. EEGMamba (arXiv 2407.20254): a different model with the same name as wang2025eegmamba; it was left out to avoid confusion, and its code was "to be released".

## Part A: candidate extra baselines

Already in refs.bib, so no new entries were added: BIOT (`yang2023biot`), BENDR (`kostas2021bendr`), ShallowConvNet/Deep4 (`schirrmeister2017deep`). They still belong in the baseline table.

| key | family | params | pretraining | code |
|---|---|---|---|---|
| altaheri2023physics (ATCNet) | compact (CNN+MHSA+TCN) | n/s (small) | none (supervised) | https://github.com/Altaheri/EEG-ATCNet |
| jiang2025neurolm (NeuroLM) | foundation (VQ tokenizer + LLM) | B 254M / L 500M / XL 1.7B | ~25,000 h | https://github.com/935963004/NeuroLM |
| chen2024eegformer (EEGFormer) | foundation (VQ transformer) | n/s (s/b/l: 6/8/12 layers) | TUH, >1.7 TB (hours n/s) | none found |
| yuan2024brainwave (BrainWave) | foundation (transformer, EEG+iEEG) | n/s (768-d, 10 layers) | 40,907 h (35,676 h EEG + 5,231 h iEEG), ~16k subjects | weights promised on publication; no official repo found |
| tegon2025femba (FEMBA) | state-space (bidirectional Mamba) FM | tiny 7.8M / base 47.7M / large 77.8M / huge 386M | ~21,000 h (TUEG) | https://github.com/pulp-bio/BioFoundation |
| zhou2025csbrain (CSBrain) | foundation (transformer, cross-scale tokens + sparse attention) | n/s (12 layers, d=200) | TUEG, >9,000 h | https://github.com/yuchen2199/CSBrain |
| elouahidi2025reve (REVE) | foundation (MAE transformer, 4D pos. enc., any montage) | small 12.7M / base 69.2M / large ~0.4B (HF collection) | >60,000 h, 92 datasets, 25k subjects | https://github.com/elouayas/reve_eeg ; https://brain-bzh.github.io/reve/ ; HF: huggingface.co/collections/brain-bzh/reve |
| doner2025luna (LUNA) | foundation (cross-attn channel compression + transformer; topology-agnostic) | base 7M / large 43M / huge 311.4M | ~21,900 h (TUEG + Siena) | https://github.com/pulp-bio/BioFoundation |
| ma2026codebrain (CodeBrain) | foundation / state-space hybrid (TF dual tokenizer + EEGSSM) | 15.17M default (3M-147M scaling study) | TUEG, ~9,246 h | https://github.com/jingyingma01/CodeBrain |
| wang2025eegmamba (EEGMamba, Wang et al.) | state-space (Mamba) FM, CBraMod codebase | n/s (not checked: ScienceDirect 403) | n/s (not checked) | https://github.com/wjq-learning/EEGMamba ; weights huggingface.co/weighting666/EEGMamba |
| barmpas2025advancing (LaBraM++) | foundation (improved LaBraM codebook transformer) | n/s | n/s | none found |

### Per-entry details
- **altaheri2023physics**: https://doi.org/10.1109/TII.2022.3197419. IEEE TII 19(2):2249-2258, Feb 2023 (online Aug 2022). ATCNet is an attention-based temporal convolutional network for motor imagery, and a strong compact supervised baseline.
- **jiang2025neurolm**: https://arxiv.org/abs/2409.00101 (journal_ref: ICLR 2025). Uses a text-aligned VQ neural tokenizer and treats EEG as a "foreign language" for an LLM, with multi-task instruction tuning.
- **chen2024eegformer**: https://arxiv.org/abs/2401.10278 (arXiv preprint only). A vector-quantized transformer pretrained on TUH, aimed at interpretable transferable representations.
- **yuan2024brainwave**: https://arxiv.org/abs/2402.10251 (arXiv v8, no venue). The first FM for both invasive and non-invasive recordings, aimed at clinical diagnosis.
- **tegon2025femba**: https://arxiv.org/abs/2502.06438 , DOI 10.1109/EMBC58623.2025.11252697 (EMBC 2025). A bidirectional Mamba that scales linearly with sequence length. A follow-up, "FEMBA on the Edge" (IEEE TBME 2026, DOI 10.1109/TBME.2026.3683482), also exists if needed.
- **zhou2025csbrain**: https://arxiv.org/abs/2506.23075 ; https://neurips.cc/virtual/2025/poster/117249 (NeurIPS 2025; the repo says Spotlight). Cross-scale spatiotemporal tokenization plus structured sparse attention, evaluated on 11 tasks across 16 datasets.
- **elouahidi2025reve**: https://arxiv.org/abs/2510.21585 ; NeurIPS 2025 proceedings at papers.nips.cc (hash 20a917f7...). The largest pretraining corpus here, with strong linear-probing results. A natural cross-montage baseline.
- **doner2025luna**: https://arxiv.org/abs/2510.22257 ; NeurIPS 2025 proceedings at papers.nips.cc (hash 66969a9e...). Compresses any electrode layout into a fixed latent with learned queries, so compute is linear in the number of channels.
- **ma2026codebrain**: https://arxiv.org/abs/2506.09110 ; ICLR 2026 (proceedings.iclr.cc, iclr.cc/virtual/2026/poster/10007532). Decoupled time/frequency tokenizer plus multi-scale SSM with sliding-window attention.
- **wang2025eegmamba**: DOI 10.1016/j.neunet.2025.107816 (Neural Networks 192:107816, 2025; authors confirmed via Crossref). A Mamba EEG FM from the CBraMod group, with public pretrained weights.
- **barmpas2025advancing**: https://arxiv.org/abs/2505.16724 (arXiv preprint only). LaBraM++ improves on LaBraM with signal-processing-motivated design changes.

## Part B: related work

- **kim2025aligning**: DOI 10.1109/SMC58881.2025.11343689 ; https://arxiv.org/abs/2507.13171. RL from implicit human feedback: a pretrained ErrP decoder turns EEG into probabilistic reward for a MuJoCo Kinova pick-and-place task, and gets close to a dense-reward baseline. **Closest prior work to our RL study.**
- **kim2026accelerating**: DOI 10.1109/BCI69045.2026.11435067 ; https://arxiv.org/abs/2511.18878. ErrP reward shaping for a 7-DoF obstacle-reaching manipulator. Uses an EEGNet decoder trained leave-one-subject-out on 12 subjects, and studies how the feedback weight affects RL acceleration. **Directly relevant to our cross-subject decoder → RL link.**
- **fidencio2022errorrelated**: DOI 10.3389/fnhum.2022.806517. A review of ErrPs as reward signals in RL-based BMIs.
- **fidencio2024generic**: DOI 10.3389/fnhum.2024.1390714. A generic (subject-independent) ErrP classifier trained on SEREEGA-simulated subjects. It reaches 62.7-72.9% balanced accuracy, comparable to leave-one-subject-out training. Relevant to cross-subject ErrP decoding.
- **fidencio2025hybrid**: DOI 10.3389/fnhum.2025.1569411. An adaptive hybrid BCI that uses ErrP and motor imagery with two RL agents. The arXiv preprint 2502.18594 has a different title ("Error-related Potential driven Reinforcement Learning for adaptive Brain-Computer Interfaces") and the same authors. Cite the journal version.
- **liu2026decoding**: https://arxiv.org/abs/2607.24806 (arXiv preprint, July 2026). ErrP decoding under unimodal, bimodal and trimodal feedback with varying congruency, using a multi-branch EEGNet with auxiliary supervision.
- **zhuang2026neurocognitive**: https://arxiv.org/abs/2603.25968 (arXiv preprint). EEG/ERP-derived cognitive reward added to RL for autonomous driving (20 participants), framed as an alternative to preference-based RLHF.
- **lee2025large**: https://proceedings.mlr.press/v267/lee25a.html (PMLR 267:32878-32888, ICML 2025). Fine-tuned EEG FMs gain only 0.9-1.2% over compact deep nets while using far more parameters. LoRA is also evaluated.
- **yang2026are**: https://openreview.net/forum?id=5Xwm8e6vbh ; https://iclr.cc/virtual/2026/poster/10011450 (ICLR 2026). Benchmarks EEG FMs against traditional and compact decoders across 6 protocols. In data-scarce settings FMs often fail to beat compact or classical decoders, and linear probing is weak. Introduces ST-EEGFormer, with code at https://github.com/LiuyinYang1101/STEEGFormer. ST-EEGFormer could also serve as a Part A baseline.

Existing keys this list does not duplicate: `liu2026eegfmcompass` (arXiv 2601.17883, now with journal_ref "National Science Review, 2026", so it may be worth updating the venue in refs.bib), `kastrati2025eegbench`, `wu2025adabrainbench`, `lu2026omnieegbench`, `xiong2025eegfmbench`.
