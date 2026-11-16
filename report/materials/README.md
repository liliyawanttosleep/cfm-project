# Report materials

Everything the report needs, and where each piece comes from. All figures and tables are regenerated from `results/` by
`python report/make_figs.py` and `python report/make_tables.py` (or `bash report/build.sh`, which also compiles the PDF).
`numbers.json` in this folder is written by `make_tables.py` and holds every value used in the tables.

## Figures (`report/figs/`, vector PDF)

| File | Where | Shows | Source data |
|---|---|---|---|
| `fig_samples.pdf` | Fig. 1, §4.1 (wrapped) | Samples from the same noise at 1 / 2 / 5 / 100 steps; reflow rows appear when the reflow models exist | `results/e4/mnist_samples.npz`; reflow grids sampled on the CPU from `results/e2/ckpt/*_reflow.pt` |
| `fig_c1_reliability.pdf` | Fig. 2, §4.1 | FID and TV across three training seeds | `results/e4/mnist_eval.json`, `results/e3/fid_s{1,2}.json`, `results/c1_reliability/kmeans_c1.json` |
| `fig_c2_solvers.pdf` | App. C.2 | κ(t), the √κ grid, FID vs NFE per solver and grid, last-step variants | `results/c2_solvers/{otcfm,icfm}_s0.json`, `t1_*_s0.json` |
| `fig_frontier.pdf` | Fig. 3, §4.4 | FID vs NFE (variants without a detectable gain left out) for every model and sampler, lower envelope | all of the above plus `results/e3/fid_s0_*.json` |
| `fig_kmeans.pdf` | App. B | Generated / real share of each k-means cluster | `results/c1_reliability/kmeans_c1.json` |
| `fig_2d_paths.pdf` | App. C | 2D paths and one-step landing points, I-CFM vs OT-CFM | `results/e1/ckpt/{icfm,otcfm}_moons_s0.pt` |

## Tables (`report/tables/`)

| File | Where | Source data |
|---|---|---|
| `tab_reliability.tex` | App. C.1 | `results/e4/mnist_eval.json`, `results/e3/fid_s{1,2}.json`, `results/c1_reliability/fid_noise.json` |
| `tab_training.tex` | Table 1, §4.3 | `results/e3/fid_s0_{_otb512,_otb2048,_tlogitnormal,_tkappa,_reflow}.json`, `results/e2/*_train.json`, `results/c5_reflow/pairs_*.json` |
| `tab_sampler.tex` | App. C | `results/c2_solvers/*_s0.json`, `t1_*_s0.json` |
| `tab_pairs.tex` | App. C | `results/c3_otbatch/pairs.json`, `results/c5_reflow/pairs_*.json` |
| `tab_literature.tex` | App. C | Published papers in `papers/` (see below) |

## Derivations

The text of Appendix A follows `docs/notes/derivations.md` (D1–D9). Numerical checks of D4, D5, D8 and D9:
`python code/c2_solvers/check_derivations.py` (CPU, about 1 minute).

| Report | Notes | Claim |
|---|---|---|
| Prop. 1, App. A.1–A.2 | D1, D2 | CFM loss = marginal loss + constant; marginal field transports p0 to p1 |
| Eq. (2), App. A.3 | D3 | One Euler step lands on the posterior mean |
| Eq. (3), App. A.4 | D4 | No pairing saves more than 2 tr Σ^½ (Gelbrich) |
| Eq. (4), App. A.5 | D5 | Step density ∝ √κ and its predicted gain G |
| App. A.6 | D6 | Training-time distribution = weight in the loss |
| App. A.7 | D7 | Straightness, one-step exactness, reflow theorems |
| Eq. (5), App. A.8 | D8 | FID bias, variance components, TV noise floor, two-sample test |
| §3.1, App. A.9 | D9 | Diffusion = CFM with Gaussian noise and independent pairs after rescaling; DDIM = Euler |

## Literature numbers (Table `tab_literature.tex`)

Checked against the PDFs in `papers/` on 2026-11-14 (unconditional CIFAR-10, FID on 50k samples):

| Number | Paper | Location |
|---|---|---|
| DDPM 3.17 (1,000 steps) | 10_Ho2020_DDPM | abstract, Table 1 |
| DDIM 13.36 (10 steps), 4.16 (100 steps) | 11_Song2021_DDIM | Table 1, η = 0 row |
| EDM 1.97 (35 NFE) | 16_Karras2022_EDM | abstract, Table 2 |
| I-CFM 3.659 (146 NFE), OT-CFM 3.577 (134 NFE), dopri5 | 23_Tong2024_OT_CFM | Table 5 (right) |
| 1-rectified flow 378, 2-rectified flow 12.21 (one Euler step) | 21_Liu2023_Rectified_Flow | Table 1 (one-step generation) |
| Consistency distillation 3.55 (1 step), 2.93 (2 steps) | 31_Song2023_Consistency_Models | Table 1 |

## Course requirements (from the course guidelines, summarised in `report/README.md`)

| Requirement | Where in the report |
|---|---|
| About 5 pages, formal paper structure (Introduction, Related Work, Methods, Experiments, Conclusion, References) | §1–5; appendix separate |
| Individual Contributions section | After §5 (**both authors must fill it in**) |
| One course method | k-means + two-sample χ² class coverage: §3.4, §4.1, App. B |
| Comparison with the state of the art | §4.4 and Table `tab_literature` (App. C) |
| Results presented visually | Figs. 1–3 (samples, seed spread and class coverage, frontier) and Table 1 in the main text; App. B–C figures and tables |
| Experimental analysis | §4.1–4.4 (short) and App. C (full) with the decision rule of §3.4 |
| Derivations supporting the method | §3 (Prop. 1, Eqs. 1–5) and App. A (full derivations) |
| Code link | Individual Contributions paragraph |

Open items before submission: fill in Individual Contributions (the only red `\pending{}` left); ask the instructor whether the
appendix is allowed and whether it counts towards the page limit. The main text (title to Individual Contributions) is about 5.6 pages.
