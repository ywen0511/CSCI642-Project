# refs.bib verification log (ErrP-Bench)

Re-verified 2026-11-12. All 74 entries checked against the web (Crossref API `https://api.crossref.org/works/<DOI>`, arXiv API, PMLR, NeurIPS/ICLR proceedings, OpenReview, JMLR, publisher pages).
Keys and key order unchanged; no entries added or removed. The file was rewritten in Google-Scholar style
(field order title, author, journal/booktitle, volume, number, pages, year, publisher/organization, doi; arXiv-only papers as
`@article` with `journal={arXiv preprint arXiv:XXXX.XXXXX}`; `@misc`+eprint entries converted accordingly).
**Global formatting change applied to all entries:** field reorder; `publisher` (journals) / `organization` (conferences) added from Crossref/venue;
venue suffixes like "({ICML})"/"({IROS})" dropped in favour of the canonical booktitle; ICML/PMLR entries use `organization={PMLR}` instead of `series`/`publisher`.

Build check: `tectonic main_measured.tex` → rc=0, no `Warning--`, no undefined citations.

| key | source URL | change (beyond the global formatting) |
|---|---|---|
| ehrlich2019feasibility | https://api.crossref.org/works/10.1007/s12369-018-0501-8 | publisher Springer |
| ehrlich2018coadaptation | https://api.crossref.org/works/10.1088/1741-2552/aae069 | publisher IOP |
| chavarriaga2010learning | https://api.crossref.org/works/10.1109/TNSRE.2010.2053387 | publisher IEEE |
| kappenman2021erpcore | https://api.crossref.org/works/10.1016/j.neuroimage.2020.117465 | publisher Elsevier |
| kueper2024eegemg | https://api.crossref.org/works/10.3389/fnhum.2024.1304311 | publisher Frontiers |
| chari2024interhri | https://api.crossref.org/works/10.24963/ijcai.2024/994 | publisher IJCAI Organization |
| chavarriaga2014errare | https://api.crossref.org/works/10.3389/fnins.2014.00208 | publisher Frontiers (pages=208 is the article number) |
| falkenstein1991effects | https://api.crossref.org/works/10.1016/0013-4694(91)90062-9 | publisher Elsevier |
| gehring1993neural | https://api.crossref.org/works/10.1111/j.1467-9280.1993.tb00586.x | publisher SAGE |
| iturrate2015teaching | https://api.crossref.org/works/10.1038/srep13893 | number={1}; publisher NPG |
| salazargomez2017correcting | https://api.crossref.org/works/10.1109/ICRA.2017.7989777 | organization IEEE |
| kim2017intrinsic | https://api.crossref.org/works/10.1038/s41598-017-17682-7 | number={1}; publisher NPG |
| xu2021accelerating | https://api.crossref.org/works/10.1016/j.neucom.2021.06.064 | publisher Elsevier |
| luo2018deep | https://api.crossref.org/works/10.1109/BIBM.2018.8621183 | organization IEEE |
| akinola2020accelerated | https://api.crossref.org/works/10.1109/ICRA40945.2020.9196566 | organization IEEE |
| spuler2015error | https://api.crossref.org/works/10.3389/fnhum.2015.00155 | title casing per Crossref; publisher |
| lawhern2018eegnet | https://api.crossref.org/works/10.1088/1741-2552/aace8c | title casing ("Brain--Computer"); publisher |
| santamariavazquez2020eeginception | https://api.crossref.org/works/10.1109/TNSRE.2020.3048106 | publisher |
| chen2024eegnex | https://api.crossref.org/works/10.1016/j.bspc.2023.105475 | title casing; publisher |
| song2023eegconformer | https://api.crossref.org/works/10.1109/TNSRE.2022.3230250 | publisher |
| zhao2024ctnet | https://api.crossref.org/works/10.1038/s41598-024-71118-7 | title casing; number={1}; publisher |
| rivet2009xdawn | https://api.crossref.org/works/10.1109/TBME.2009.2012869 | publisher |
| barachant2012multiclass | https://api.crossref.org/works/10.1109/TBME.2011.2172210 | `{R}iemannian`→`{Riemannian}`; publisher |
| barachant2014plug | https://arxiv.org/abs/1409.0107 | @misc → arXiv @article (no published version found) |
| he2020transfer | https://api.crossref.org/works/10.1109/TBME.2019.2913914 | `{Euclidean}`; publisher |
| zanini2018transfer | https://api.crossref.org/works/10.1109/TBME.2017.2742541 | `{Riemannian}`; publisher |
| jiang2024labram | https://proceedings.iclr.cc/paper_files/paper/2024/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html | booktitle "International Conference on Learning Representations"; pages 16405--16426 (proceedings meta tags); url dropped |
| wang2024eegpt | https://api.crossref.org/works/10.52202/079017-1239 | none (Crossref misspells "Guagnyu"; kept "Guangyu") |
| wang2025cbramod | https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbbd6d915cb90be21c1254a82d45cedd-Abstract-Conference.html | booktitle ICLR; pages 75310--75346; url dropped |
| yang2023biot | https://api.crossref.org/works/10.52202/075280-3420 | none |
| kostas2021bendr | https://api.crossref.org/works/10.3389/fnhum.2021.653659 | publisher |
| jayaram2018moabb | https://api.crossref.org/works/10.1088/1741-2552/aadea0 | publisher |
| chevallier2024moabbbenchmark | http://export.arxiv.org/api/query?id_list=2404.15319 | @misc → arXiv @article (no journal version found) |
| kastrati2025eegbench | http://export.arxiv.org/api/query?id_list=2512.08959 ; https://neurips.cc/virtual/2025/132688 | @misc → arXiv @article; workshop moved to `note` |
| lu2026omnieegbench | http://export.arxiv.org/api/query?id_list=2606.00815 | @misc → arXiv @article |
| liu2026eegfmcompass | https://api.crossref.org/works/10.1093/nsr/nwag466 | publisher OUP; still no volume/issue in Crossref (FLAG: add when assigned) |
| kokate2026channel | https://arxiv.org/abs/2604.23091 | @misc → arXiv @article |
| wu2025adabrainbench | http://export.arxiv.org/api/query?id_list=2507.09882 | @misc → arXiv @article |
| xiong2025eegfmbench | https://proceedings.mlr.press/v306/xiong26f.html | **arXiv → ICML 2026 (PMLR 306, pp. 140324--140359); year 2025→2026** (key unchanged) |
| schirrmeister2017deep | https://api.crossref.org/works/10.1002/hbm.23730 | publisher Wiley |
| gramfort2013mne | https://api.crossref.org/works/10.3389/fnins.2013.00267 | publisher (Crossref lists only first author; 11-author list kept, matches paper) |
| haarnoja2018sac | https://proceedings.mlr.press/v80/haarnoja18b.html | booktitle ICML; organization PMLR |
| plappert2018multigoal | http://export.arxiv.org/api/query?id_list=1802.09464 | @misc → arXiv @article |
| delazcano2024gymnasiumrobotics | https://raw.githubusercontent.com/Farama-Foundation/Gymnasium-Robotics/main/README.md | url → howpublished={\url{}} (software @misc, v1.4.0, 2024) |
| raffin2021sb3 | https://jmlr.org/papers/v22/20-1364.html | `{Stable-Baselines3}` braces |
| todorov2012mujoco | https://api.crossref.org/works/10.1109/IROS.2012.6386109 | booktitle per Crossref; organization IEEE |
| ng1999policy | https://mlanthology.org/icml/1999/ng1999icml-policy/ | organization Morgan Kaufmann |
| knox2009tamer | https://api.crossref.org/works/10.1145/1597735.1597738 | organization ACM |
| christiano2017deep | https://papers.nips.cc/paper_files/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html | none |
| andrychowicz2017her | https://papers.nips.cc/paper_files/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html | none |
| demsar2006statistical | https://www.jmlr.org/papers/v7/demsar06a.html | none |
| guo2017calibration | https://proceedings.mlr.press/v70/guo17a.html | booktitle ICML; organization PMLR |
| green1966signal | standard library record (Wiley, 1966); no online page fetched | address dropped (FLAG: not web-fetched, classic book) |
| hanley1982meaning | https://api.crossref.org/works/10.1148/radiology.143.1.7063747 | publisher RSNA |
| altaheri2023physics | https://api.crossref.org/works/10.1109/TII.2022.3197419 | publisher IEEE |
| jiang2025neurolm | https://openreview.net/forum?id=Io9yFt7XH7 ; arXiv 2409.00101 journal_ref | booktitle ICLR; url arXiv → OpenReview forum |
| chen2024eegformer | http://export.arxiv.org/api/query?id_list=2401.10278 | none (still arXiv-only) |
| yuan2024brainwave | http://export.arxiv.org/api/query?id_list=2402.10251 | none (still arXiv-only) |
| tegon2025femba | Crossref DOI 10.1109/EMBC58623.2025.11252697 | booktitle "2025 ..." per Crossref; organization IEEE |
| zhou2025csbrain | https://api.crossref.org/works/10.52202/085713-2915 | **NeurIPS 2025 published: volume 38, pages 96859--96904, doi added; arXiv url removed** |
| elouahidi2025reve | https://api.crossref.org/works/10.52202/085713-0760 | **volume 38, pages 25583--25619, doi added; url removed** |
| doner2025luna | https://api.crossref.org/works/10.52202/085713-2377 | **volume 38, pages 78772--78798, doi added; url removed** |
| ma2026codebrain | https://openreview.net/forum?id=msJgEkjwh5 | booktitle ICLR; url arXiv → OpenReview forum |
| wang2025eegmamba | https://api.crossref.org/works/10.1016/j.neunet.2025.107816 | publisher Elsevier |
| barmpas2025advancing | http://export.arxiv.org/abs/2505.16724 | none (arXiv-only; NeurIPS'25 BrainBodyFM workshop poster is non-archival) |
| kim2025aligning | https://api.crossref.org/works/10.1109/SMC58881.2025.11343689 | booktitle "2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC)"; organization IEEE |
| kim2026accelerating | https://api.crossref.org/works/10.1109/BCI69045.2026.11435067 | booktitle "2026 14th International Conference on Brain-Computer Interface (BCI)"; organization IEEE |
| fidencio2022errorrelated | https://api.crossref.org/works/10.3389/fnhum.2022.806517 | publisher |
| fidencio2024generic | https://api.crossref.org/works/10.3389/fnhum.2024.1390714 | publisher |
| fidencio2025hybrid | https://api.crossref.org/works/10.3389/fnhum.2025.1569411 | publisher |
| liu2026decoding | http://export.arxiv.org/abs/2607.24806 | none (arXiv-only) |
| zhuang2026neurocognitive | https://openaccess.thecvf.com/content/CVPR2026/html/Zhuang_Neuro-Cognitive_Reward_Modeling_for_Human-Centered_Autonomous_Vehicle_Control_CVPR_2026_paper.html ; https://cvpr.thecvf.com/virtual/2026/poster/37908 | **arXiv → @inproceedings CVPR 2026, pages 10599--10609** (no DOI) |
| lee2025large | https://proceedings.mlr.press/v267/lee25a.html | booktitle ICML; organization PMLR |
| yang2026are | https://openreview.net/forum?id=5Xwm8e6vbh | booktitle ICLR |

## Flags
- **green1966signal:** a classic book that was not fetched online. The publisher (Wiley) and year (1966) are the standard catalog record.
- **liu2026eegfmcompass:** published online only. Crossref has no volume or issue yet, so the article number nwag466 stands in for pages.
- **xiong2025eegfmbench:** the key says 2025 but the year is now 2026 (ICML 2026). The key was not changed.
- **ICLR entries:** `jiang2024labram` and `wang2025cbramod` carry page numbers taken from proceedings.iclr.cc. `jiang2025neurolm`, `ma2026codebrain` and `yang2026are` keep an OpenReview `url` because they have no DOI or pages.
- **Unverifiable entries:** none. Every entry is a real publication.

## Duplicates
None. Related but distinct: `ehrlich2018coadaptation` vs `ehrlich2019feasibility`; `barmpas2025advancing` vs `lee2025large`; `fidencio2022/2024/2025`.
