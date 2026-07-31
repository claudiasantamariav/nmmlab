# Rosetta Stone of Neural Mass Models

**Live site: [claudiasantamariav.github.io/nmmlab](https://claudiasantamariav.github.io/nmmlab/)**

This repo hosts a companion website and Python package for *Rosetta Stone of Neural Mass Models*, a paper proposing a unifying "ladder" for neural mass models (NMMs) — the equations used to describe how populations of neurons generate the oscillations seen in EEG, MEG, and fMRI.

The core idea: instead of treating each NMM formalism (lumped-parameter models, firing-rate equations, multi-layer generators, ...) as its own island, start from the simplest possible oscillator — a coupled excitatory/inhibitory push-pull pair — and climb a systematic ladder of biological detail. Each rung is presented unforced, then forced, then as a coupled network, mirroring the move from single-node to whole-brain modeling. The website is an interactive companion to that argument: you can read the derivation, then immediately play with the model that was just derived.

## What's in the repo

| Path | What it is |
|---|---|
| `docs/` | The website itself (static site, served via GitHub Pages from `main`/`docs`) |
| `nmmlab/` | Python package implementing the five models covered by the paper |
| `notebooks/` | Jupyter notebooks used to explore/simulate each model |
| `paper_source/` | LaTeX source for the paper and appendix (also rendered on the site) |

## The website

The site is the "ladder" made interactive — six main pages plus the full paper:

- **Home** (`index.html`) — the ladder overview, links into each model
- **Tutorial** (`tutorial.html`) — intro framing for readers new to NMMs
- **The Ladder** (`map.html`) — the full derivation, rung by rung, from harmonic oscillator to network models
- **Demos** (`explorer.html`) — interactive simulators for each model, with sliders for the physical parameters
- **Compare** (`compare.html`) — side-by-side comparison across models
- **Oscillation** (`oscillation.html`) — a deep-dive on what "an oscillation" actually means dynamically
- **Paper / Appendix** (`paper.html`, `paper-appendix.html`) — the full paper, converted from the LaTeX source and wired up with in-page citations and links to the relevant demo for each derivation

### How it's built

It's a static site with **no build step, no bundler, no framework, and no CI** — just hand-written HTML/CSS/JS pages, opened directly in a browser or served as-is by GitHub Pages.

- Each page is mostly self-contained (inline `<style>`/`<script>`), so any page can be read and edited on its own.
- Interactive plots (sliders driving live simulations of each NMM) are built on **Plotly** (`docs/plotly.min.js`, vendored rather than pulled from a CDN, so the site works offline and doesn't depend on external uptime).
- Mathematical notation is rendered with **KaTeX** (`docs/assets/katex/`), also vendored.
- Figures used in the paper pages live under `docs/assets/paper/`.
- `paper.html` and `paper-appendix.html` were produced by converting the LaTeX in `paper_source/` to HTML and then hand-linking each derivation to its corresponding interactive demo — the site isn't just the paper as a PDF-in-a-browser, it's meant to be read alongside the simulators.
- Deployment is just GitHub Pages pointed at `/docs` on `main` — pushing to `main` is the entire release process.

### Local development

No install needed — clone the repo and open any file under `docs/` directly in a browser, or serve the folder locally:

```bash
cd docs
python -m http.server 8000
# visit http://localhost:8000
```

## The `nmmlab` Python package

The package implements simulation code for each model on the ladder, one module per model. Every model follows the same pattern: an unforced version, a forced version, and a network version, matching the site's Demos page.

| Module | Model |
|---|---|
| `ho.py` | Harmonic oscillator (`ho`, `forced_dho`, `coupled_dho`, `kuramoto`, `resonance_curve`) |
| `stuart_landau.py` | Stuart-Landau oscillator (`sl`, `sl_forced`, `sl_network`, `bifurcation_curve`) |
| `wilson_cowan.py` | Wilson-Cowan (`wilco_unforced`, `wilco_forced`, `wilco_network`) |
| `jansen_rit.py` | Jansen-Rit / NMM1 (`nmm1`, `nmm1_network`, `k_crit`) |
| `montbrio.py` | Montbrio-Pazo-Roxin / NMM2 (`mpr`, `nmm2`, `nmm2_network`) |

Stochastic models are integrated with an Euler-Maruyama scheme (`ho.euler_maruyama`); dependencies are just `numpy`, `scipy`, and `matplotlib`.

### Install

```bash
pip install -e .
```

### Usage

```python
import numpy as np
from nmmlab import ho

t = np.linspace(0, 50, 5000)
z = ho(omega=1.0, alpha=-0.1, z0=1 + 0j, t=t)  # damped harmonic oscillator
```

The `notebooks/` folder has a runnable simulation notebook per model (`HO simulation.ipynb`, `SL simulation.ipynb`, `WILCO simulation.ipynb`, `NMM1 simulation.ipynb`) showing typical parameter choices and plots.

## Citation

Castaldo, F.\*, de Palma Aristides, R., Clusella, P., Garcia-Ojalvo, J., & Ruffini, G.\* (2025). *Rosetta Stone of Neural Mass Models*. (\*equal contribution)

The full paper and appendix are readable on the site at [`paper.html`](https://claudiasantamariav.github.io/nmmlab/paper.html) / [`paper-appendix.html`](https://claudiasantamariav.github.io/nmmlab/paper-appendix.html), with LaTeX source in `paper_source/`.
