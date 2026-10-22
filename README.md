# natura2000-laganas-posidonia-evidence

[![CI](https://github.com/annefou/natura2000-laganas-posidonia-evidence/actions/workflows/ci.yml/badge.svg)](https://github.com/annefou/natura2000-laganas-posidonia-evidence/actions/workflows/ci.yml)
[![Jupyter Book](https://github.com/annefou/natura2000-laganas-posidonia-evidence/actions/workflows/jupyter-book.yml/badge.svg)](https://annefou.github.io/natura2000-laganas-posidonia-evidence/)
[![Docker](https://github.com/annefou/natura2000-laganas-posidonia-evidence/actions/workflows/docker.yml/badge.svg)](https://github.com/annefou/natura2000-laganas-posidonia-evidence/pkgs/container/natura2000-laganas-posidonia-evidence)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/{{ZENODO_DOI}}.svg)]({{ZENODO_DOI}})
[![FAIR4RS](https://img.shields.io/badge/FAIR4RS-conformant-brightgreen)](docs/fair4rs-checklist.md)
[![FORRT](https://img.shields.io/badge/FORRT-replication-blue)](https://forrt.org/)
[![Science Live](https://img.shields.io/badge/Science%20Live-nanopub%20chain-purple)](nanopubs/PUBLISHED.md)
[![RO-Crate](https://img.shields.io/badge/RO--Crate-1.2-orange)](ro-crate-metadata.json)
[![Software Heritage](https://archive.softwareheritage.org/badge/origin/https://github.com/annefou/natura2000-laganas-posidonia-evidence/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/annefou/natura2000-laganas-posidonia-evidence)

> **Towards Global-Scale Seagrass Mapping and Monitoring Using Sentinel-2 on Google Earth Engine: The Case Study of the Aegean and Ionian Seas** — replication study.
> Reference paper: [10.3390/rs10081227](https://doi.org/10.3390/rs10081227)

**DRAFT, not reviewed.** Part of LifeWatch ERIC's country evidence for Greece. Two analyses for *Posidonia oceanica* meadows (habitat 1120*) in Natura 2000 site GR2210002, Laganas Bay (Zakynthos):

1. **Extent accuracy** (`pipeline/accuracy/`). A Sentinel-2 summer composite, glint correction, depth-invariant indices and a Random Forest, validated by spatial block hold-out (1 km blocks, five folds) against the HCMR habitat map of the site (1999–2002, EMODnet Seabed Habitats GR000019) and checked against the 2015 seagrass survey (Panayotidis et al. 2022, doi:10.17882/87740). Includes a 2024 vs 2025 stability test and aggregation to HEALPix cells.
2. **Marine heatwave exposure** (`pipeline/climate/`). Copernicus Marine sea-surface temperature and Mediterranean reanalysis at meadow depth, 1982 to date, marine heatwaves after Hobday et al. (2016), exceedance of published Posidonia heat thresholds per meadow cell, and projections from the DestinE Climate Change Adaptation Digital Twin (IFS-NEMO, IFS-FESOM; historical then SSP3-7.0), compared with observations and bias-corrected first.

It measures exposure, not impact: whether meadows are harmed needs field condition data.

**Grid.** Results are published on HEALPix cells with [healpix-geo](https://github.com/GRID4EARTH/healpix-geo), WGS84 ellipsoid, NESTED indexing. The 10 m UTM grid is only a working grid. The spherical HEALPix definition is used for one thing only: to address the Climate DT's own grid, which is defined on the sphere.

---

## Quick start

```bash
git clone https://github.com/annefou/natura2000-laganas-posidonia-evidence.git
cd natura2000-laganas-posidonia-evidence
pixi install
pixi run accuracy            # open data, no account (~1.5 GB of Sentinel-2 reads)
pixi run climate-observed    # needs a free Copernicus Marine account (copernicusmarine login)
pixi run climate-dt          # needs DestinE upgraded access (~/.polytopeapirc)
```

Work files go to `work/` (gitignored). Raw Climate DT extracts stay in `work/climate/data/dt_raw/`: the Climate DT terms restrict redistribution, so only derived indicators are reported.

**Inputs and licences.** Natura 2000 boundary (EEA). HCMR habitat map GR000019 (CC BY 4.0). 2015 seagrass survey, SEANOE archive of the EMODnet layer (CC BY 4.0); inside the site it matches the EMODnet WFS layer to within 1 ha of 1,748 ha. EMODnet Bathymetry DTM (CC BY 4.0). Sentinel-2 L2A from Earth Search (Copernicus terms). Copernicus Marine products (Copernicus Marine licence). DestinE Climate DT (DestinE terms).

(Pixi resolves `pixi.toml` against the per-platform `pixi.lock`, installs the env under `.pixi/`, and provides `pixi run` for any task without needing an `activate` step.)

Or with Docker:

```bash
docker run --rm ghcr.io/annefou/natura2000-laganas-posidonia-evidence:latest
```

The Jupyter Book version is at <https://annefou.github.io/natura2000-laganas-posidonia-evidence/>.

## Built from a template

This repository was created from [`sciencelivehub/forrt-replication-template`](https://github.com/sciencelivehub/forrt-replication-template). The template ships an operating manual for AI assistants ([`CLAUDE.md`](CLAUDE.md), [`AGENTS.md`](AGENTS.md)), domain conventions ([`DOMAIN.md`](DOMAIN.md)), and reference docs (`docs/`) so that an AI working only inside this repository can guide a researcher from "paper PDF + GitHub repo" to "published FORRT chain + Zenodo DOI" with no other context.

If you are reading this in a fresh fork, run [`/init-template`](.claude/skills/init-template/SKILL.md) inside Claude Code to substitute the placeholder tokens with your details. (For other AI tools, see [`docs/ai-portability.md`](docs/ai-portability.md).)

After `/init-template`, do these one-time setup steps to enable the full CI/CD path:

- **Enable GitHub Pages** at *Settings → Pages → Source: GitHub Actions*. Until enabled, the Jupyter Book build runs but the deploy step is skipped (CI stays green).
- All three workflows share one **readiness guard** (`.github/actions/check-ready`). Before `/init-template` runs, the `.template-uninitialised` sentinel makes them skip with an informative `::notice::` (badges stay green); `/init-template` deletes the sentinel, which activates them. They also skip while `notebooks/*.py` are still scaffolds (Phase 2). **Once you've published a nanopub chain** (real URIs in `nanopubs/PUBLISHED.md`), a skip is treated as a bug and **fails the run loudly** — so a finished replication can't sit on silently-green-but-empty CI.

## Repository structure

```
.
├── CLAUDE.md / AGENTS.md       # operating manual for AI assistants
├── DOMAIN.md                   # domain flavour (current: biodiversity + earth observation)
├── USER_PREFERENCES.md         # per-user style (edit on first clone)
├── README.md                   # this file
├── LICENSE                     # MIT
├── CITATION.cff                # how to cite
├── codemeta.json               # software metadata (CodeMeta-2.0)
├── ro-crate-metadata.json      # research object packaging (RO-Crate 1.2)
├── pixi.toml + pixi.lock       # pinned dependencies (single source of truth; lockfile is per-platform)
├── Dockerfile                  # container build
├── Snakefile                   # pipeline orchestration
├── myst.yml + index.md         # Jupyter Book scaffold
├── paper/                      # the source paper PDF
├── data/                       # downloaded artefacts (gitignored)
├── notebooks/                  # jupytext .py pipeline (01–04)
├── nanopubs/                   # FORRT chain drafts + published-URI registry
├── docs/                       # reference material
├── figures/                    # curated figures used in the Jupyter Book
├── .github/workflows/          # CI, Jupyter Book, Docker
└── .claude/                    # Claude Code agents, skills, sandbox config
```

## What you get

This template bakes in conventions that took multiple replications to discover. By using it, you inherit:

- **FAIR4RS conformance** — see [`docs/fair4rs-checklist.md`](docs/fair4rs-checklist.md) for the principle-by-principle mapping.
- **Self-contained data downloads** — the first notebook fetches everything; no manual data prep.
- **`pixi.toml` + `pixi.lock` as single source of truth** — local dev, Docker, and CI all install the same per-platform-pinned env.
- **`prefix-dev/setup-pixi`-based CI** — caches the env, runs the pipeline with `pixi run`, executes notebooks via a glob, fails fast on a stale lockfile.
- **Jupyter Book deployment** — auto-deploys to GitHub Pages with `BASE_URL` set correctly. (Don't put `base_url` in `myst.yml` — MyST silently ignores it.)
- **Docker + GHCR + Zenodo image archival** — `release` trigger pushes to GHCR and (optionally) archives to Zenodo for long-term preservation.
- **RO-Crate packaging** — the entire repo is a navigable Research Object via `ro-crate-metadata.json` (Process Run Crate + Workflow RO-Crate profiles).
- **Six-step FORRT chain workspace** — `nanopubs/drafts/` has a field-by-field skeleton for each step. `nanopubs/PUBLISHED.md` is the URI registry.
- **Layered AI guidance** — `CLAUDE.md` (universal) + `DOMAIN.md` (swappable per field) + `USER_PREFERENCES.md` (per-user). See [`docs/ai-portability.md`](docs/ai-portability.md) for non-Claude AI tools.
- **Sandbox by default** — `.claude/settings.json` denies file ops outside the repo, so a fresh AI session can't accidentally read `~/.ssh/` or write to `/etc/`.

## The six FORRT chain steps

A complete FORRT chain has six steps published on [platform.sciencelive4all.org](https://platform.sciencelive4all.org):

```
Quote-with-comment  →  AIDA  →  FORRT Claim  →  Replication Study  →  Replication Outcome  →  CiTO Citation
```

(For question-rooted chains with no upstream paper, replace step 1 with PICO or PCC. See [`docs/chain-decision-tree.md`](docs/chain-decision-tree.md).)

Drafts live in [`nanopubs/drafts/`](nanopubs/drafts/) field-by-field. Published URIs go into [`nanopubs/PUBLISHED.md`](nanopubs/PUBLISHED.md).

Optional further layers:

- **Research Software nanopub** — for reusable upstream tools (not demo repos). See [`docs/forrt-form-fields.md`](docs/forrt-form-fields.md) § Research Software.
- **Research Synthesis nanopub** — when this chain is part of a multi-chain story. See [`docs/forrt-form-fields.md`](docs/forrt-form-fields.md) § Research Synthesis.

## After publishing

When the chain is live and the FAIR4RS checklist is green, drafting an announcement post is the next step. See [`docs/announcement-template.md`](docs/announcement-template.md) for the structural template (vision-piece-first; the worked replication is the payoff, not the lead).

For lower-level nanopub work — retraction, superseding, batch publishing — see [`docs/programmatic-nanopubs.md`](docs/programmatic-nanopubs.md).

## Citation

If you use this work, please cite both:

- This software: [`CITATION.cff`](CITATION.cff) → DOI [{{ZENODO_DOI}}]({{ZENODO_DOI}})
- The original paper: [10.3390/rs10081227](https://doi.org/10.3390/rs10081227)

## Acknowledgements

This repository was built from [`sciencelivehub/forrt-replication-template`](https://github.com/sciencelivehub/forrt-replication-template), part of the [Science Live platform](https://platform.sciencelive4all.org). The template is licensed MIT and contributions (especially new domain flavours under [`docs/domain-flavours/`](docs/domain-flavours/)) are welcome.
