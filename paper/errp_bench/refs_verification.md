# refs.bib verification log (ErrP-Bench)

Checked 2026-11-12. For each key: the source used to verify authors, title, venue, year, volume/pages/DOI, and a one-line note on what the work contributes. Every one of the 54 entries in `refs.bib` was matched against the source below. Where a DOI exists, `https://doi.org/<DOI>` resolves to the publisher page. Crossref metadata was fetched from `https://api.crossref.org/works/<DOI>`.

## Datasets / ErrP background

| key | verification URL | what to cite it for |
|---|---|---|
| ehrlich2019feasibility | https://doi.org/10.1007/s12369-018-0501-8 (+ README of github.com/stefan-ehrlich/dataset-ErrP-HRI, which names this paper as the dataset's source) | The ErrP-HRI dataset: EEG recorded while subjects watched correct or erroneous cursor and robot actions; a feasibility test for validating robot actions with ErrPs. Int J Soc Robot 11(2), 2019 (online Nov 2018). |
| ehrlich2018coadaptation | https://doi.org/10.1088/1741-2552/aae069 | Same authors: closed-loop human–agent co-adaptation driven by ErrPs (J Neural Eng 2018). Optional related work, not the dataset paper. |
| chavarriaga2010learning | https://doi.org/10.1109/TNSRE.2010.2053387 | Observers' ErrPs used as a learning signal (reward) for an agent; source of the monitoring-ErrP data distributed as BNCI Horizon 013-2015. |
| kappenman2021erpcore | https://doi.org/10.1016/j.neuroimage.2020.117465 | ERP CORE: open dataset of 7 ERP paradigms (incl. ERN in a flanker task), 40 subjects, with standardized pipelines. NeuroImage 225:117465. |
| kueper2024eegemg | https://doi.org/10.3389/fnhum.2024.1304311 | EEG+EMG dataset of errors introduced by an active orthosis; the data behind IntEr-HRI. |
| chari2024interhri | https://doi.org/10.24963/ijcai.2024/994 | The IntEr-HRI competition (IJCAI 2024, pp. 8623–8626): asynchronous ErrP detection during orthosis-based HRI. |
| chavarriaga2014errare | https://doi.org/10.3389/fnins.2014.00208 | Review of ErrPs in BMIs: types (response, feedback, observation, interaction ErrP) and their applications. Frontiers in Neuroscience 8:208. |
| falkenstein1991effects | https://doi.org/10.1016/0013-4694(91)90062-9 | One of the two papers that discovered the ERN ("Ne"): error processing in choice-reaction tasks (Part II). |
| gehring1993neural | https://doi.org/10.1111/j.1467-9280.1993.tb00586.x | The other ERN discovery paper; it named the error-related negativity. |
| iturrate2015teaching | https://doi.org/10.1038/srep13893 | ErrPs as reward that teaches a BMI/agent the user's goal (RL from ErrPs), instead of decoding movement directly. |
| salazargomez2017correcting | https://doi.org/10.1109/ICRA.2017.7989777 | Real-time closed-loop correction of a robot's binary choice from observers' ErrPs (Baxter, ICRA 2017). |
| kim2017intrinsic | https://doi.org/10.1038/s41598-017-17682-7 | ErrPs as intrinsic implicit reward for RL on a real robot (gesture-to-action mapping). Sci Rep 7:17562. |
| xu2021accelerating | https://doi.org/10.1016/j.neucom.2021.06.064 | EEG ErrP-based implicit human feedback accelerates deep RL in game environments. Neurocomputing 460. |
| luo2018deep | https://doi.org/10.1109/BIBM.2018.8621183 | Deep RL trained from ErrPs through an EEG BCI (BIBM 2018). This is what turned up for "Luo et al. 2018"; see notes. |
| akinola2020accelerated | https://doi.org/10.1109/ICRA40945.2020.9196566 ; arXiv https://arxiv.org/abs/1910.00682 | Accelerated robot learning: an ErrP-decoded sparse human signal shapes a policy that then bootstraps RL (ICRA 2020). |
| spuler2015error | https://doi.org/10.3389/fnhum.2015.00155 | ErrPs during continuous (not discrete) feedback; detects errors of different type and severity. Front Hum Neurosci 9:155. |

## Decoders

| key | verification URL | what to cite it for |
|---|---|---|
| lawhern2018eegnet | https://doi.org/10.1088/1741-2552/aace8c | EEGNet: compact depthwise/separable CNN for EEG BCIs (including ERN/ErrP). J Neural Eng 15(5):056013. |
| santamariavazquez2020eeginception | https://doi.org/10.1109/TNSRE.2020.3048106 | EEG-Inception: inception-style CNN for ERP-based BCIs. |
| chen2024eegnex | https://doi.org/10.1016/j.bspc.2023.105475 | EEGNeX: improved EEGNet-style CNN, plus a benchmark of EEG decoders. BSPC 87:105475 (2024). |
| song2023eegconformer | https://doi.org/10.1109/TNSRE.2022.3230250 | EEG Conformer: a CNN front end plus a transformer encoder. |
| zhao2024ctnet | https://doi.org/10.1038/s41598-024-71118-7 | CTNet: convolutional transformer for motor imagery classification. Sci Rep 14:20237. |
| rivet2009xdawn | https://doi.org/10.1109/TBME.2009.2012869 | xDAWN spatial filtering to enhance evoked potentials; the basis of the xDAWN+Riemannian ERP pipeline. |
| barachant2012multiclass | https://doi.org/10.1109/TBME.2011.2172210 | Riemannian geometry classification of covariance matrices (MDM and tangent-space mapping). |
| barachant2014plug | https://arxiv.org/abs/1409.0107 | ERP "super-trial" covariances with Riemannian classification; calibration-free P300 (Brain Invaders). |
| he2020transfer | https://doi.org/10.1109/TBME.2019.2913914 | Euclidean Alignment (EA): whitening each subject by its mean covariance, for cross-subject transfer. |
| zanini2018transfer | https://doi.org/10.1109/TBME.2017.2742541 | Riemannian re-centering of each session/subject's covariances to the identity, for transfer learning. |

## Foundation models / benchmarks / tooling

| key | verification URL | what to cite it for |
|---|---|---|
| jiang2024labram | https://proceedings.iclr.cc/paper_files/paper/2024/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html ; https://arxiv.org/abs/2405.18765 ; OpenReview id QzTpTRVtrP | LaBraM: channel-patch EEG foundation model, pretrained on ~2.5k h by masked neural-code prediction (ICLR 2024). |
| wang2024eegpt | https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html | EEGPT: ~10M-param pretrained transformer with dual self-supervision (masked reconstruction + spatio-temporal alignment), evaluated by linear probing. NeurIPS 37, pp. 39249–39280. The proceedings page misspells the first author as "Guagnyu"; the .bib uses "Guangyu". |
| wang2025cbramod | https://arxiv.org/abs/2412.07236 (states ICLR 2025 acceptance); https://iclr.cc/virtual/2025/poster/29880 ; OpenReview id NPNUHgHF2w | CBraMod: criss-cross (parallel spatial/temporal) attention transformer, pretrained on TUEG by masked patch reconstruction. |
| yang2023biot | https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html | BIOT: biosignal transformer that tokenizes channels with different counts and lengths into one sequence, for cross-dataset learning. NeurIPS 36. The arXiv title differs ("BIOT: Cross-data Biosignal Learning in the Wild"). |
| kostas2021bendr | https://doi.org/10.3389/fnhum.2021.653659 | BENDR: wav2vec-2.0-style contrastive self-supervised pretraining on large amounts of EEG. |
| jayaram2018moabb | https://doi.org/10.1088/1741-2552/aadea0 | MOABB: an open framework for reproducible BCI algorithm benchmarking. |
| chevallier2024moabbbenchmark | https://arxiv.org/abs/2404.15319 | The large MOABB benchmark (many pipelines × datasets across MI/ERP/SSVEP). Note: the first author is Chevallier, not Aristimunha. |
| kastrati2025eegbench | https://arxiv.org/abs/2512.08959 | EEG-Bench: benchmark of EEG foundation models on clinical tasks (BrainBodyFM workshop @ NeurIPS). |
| lu2026omnieegbench | https://arxiv.org/abs/2606.00815 | OmniEEG-Bench: standardized evaluation benchmark for EEG foundation models. |
| liu2026eegfmcompass | https://doi.org/10.1093/nsr/nwag466 | EEG-FM-Compass: review plus benchmark of EEG foundation models (Natl Sci Rev 2026, article nwag466; no volume/issue assigned yet). |
| kokate2026channel | https://arxiv.org/abs/2604.23091 | Systematic benchmark of channel-adaptation strategies for EEG foundation models across architectures, tasks and training regimes. |
| wu2025adabrainbench | https://arxiv.org/abs/2507.09882 | AdaBrain-Bench: 13 datasets / 7 BCI tasks; cross-subject, multi-subject and few-shot evaluation of LaBraM, CBraMod, EEGPT, BIOT and others. |
| xiong2025eegfmbench | https://arxiv.org/abs/2508.17742 | EEG-FM-Bench: 14 datasets / 10 paradigms; reports that frozen-backbone FM features generalize poorly. arXiv v2 says "Accepted by ICML2026". Update the entry to PMLR once the proceedings page exists. |
| schirrmeister2017deep | https://doi.org/10.1002/hbm.23730 | ShallowConvNet/DeepConvNet; the basis of braindecode. Hum Brain Mapp 38(11). |
| gramfort2013mne | https://doi.org/10.3389/fnins.2013.00267 ; Europe PMC PMID 24431986 | MNE-Python software for EEG/MEG processing. |

## RL

| key | verification URL | what to cite it for |
|---|---|---|
| haarnoja2018sac | https://proceedings.mlr.press/v80/haarnoja18b.html | Soft Actor-Critic (max-entropy off-policy RL). PMLR 80:1861–1870. |
| plappert2018multigoal | https://arxiv.org/abs/1802.09464 | Introduces the Fetch/ShadowHand goal-conditioned robotics tasks. |
| delazcano2024gymnasiumrobotics | https://github.com/Farama-Foundation/Gymnasium-Robotics (README BibTeX) | The Gymnasium Robotics software package (maintained Fetch environments). README gives v1.4.0 / 2024; change the version and year if you used a different release. |
| raffin2021sb3 | https://www.jmlr.org/papers/v22/20-1364.html | Stable-Baselines3 RL library. JMLR 22(268):1–8. |
| todorov2012mujoco | https://doi.org/10.1109/IROS.2012.6386109 | The MuJoCo physics engine. |
| ng1999policy | https://mlanthology.org/icml/1999/ng1999icml-policy/ ; https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html | Potential-based reward shaping preserves optimal policies. ICML 1999, pp. 278–287 (no DOI). |
| knox2009tamer | https://doi.org/10.1145/1597735.1597738 | TAMER: learning from scalar human reinforcement signals. K-CAP 2009. |
| christiano2017deep | https://papers.nips.cc/paper_files/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html | RL from human preference comparisons via a learned reward model. |
| andrychowicz2017her | https://papers.nips.cc/paper_files/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html | Hindsight Experience Replay for sparse-reward goal-conditioned tasks (Fetch). |

## Stats / metrics

| key | verification URL | what to cite it for |
|---|---|---|
| demsar2006statistical | https://jmlr.org/papers/v7/demsar06a.html | Friedman/Nemenyi tests and critical-difference diagrams for comparing many classifiers over many datasets. |
| guo2017calibration | https://proceedings.mlr.press/v70/guo17a.html | Miscalibration of modern networks; ECE and temperature scaling. |
| green1966signal | https://libcat.library.tamu.edu/Record/in00000109178 (library catalog; no publisher page found) | Classical signal detection theory; equal-variance binormal model linking d′ and AUC (AUC = Φ(d′/√2)). |
| hanley1982meaning | https://doi.org/10.1148/radiology.143.1.7063747 | Meaning of AUROC as P(score_pos > score_neg) and its standard error. |

## Notes / corrections to the request

- "Ehrlich & Cheng 2019": the dataset paper is the Int J Social Robotics one (not Frontiers in Neurorobotics), as confirmed by the GitHub README.
- "Luo et al. 2018": no 2018 paper by Luo with the title "Accelerated robot learning via human brain signals" exists. That title belongs to Akinola et al. (arXiv 2019, ICRA 2020). The 2018 Luo paper found is Luo, Fan, Lv, Zhou, BIBM 2018 (`luo2018deep`). Confirm it is the work you meant.
- Falkenstein 1991: cited as Part II (error processing), the ERN paper. Part I (DOI 10.1016/0013-4694(91)90061-8) covers simple/choice RT and is not included.
- Barachant 2012/2013: only the TBME 2012 paper was requested and included. The 2013 Neurocomputing tangent-space-classifier paper was not added.
- Crossref gives only initials for Falkenstein et al. (kept as initials) and for Hanley & McNeil. For Hanley & McNeil the .bib uses the well-known full names James A. / Barbara J.
