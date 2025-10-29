# Forecasting Faculty Placement from Co-authorship, Bibliometrics, and CV Features

This repository contains the end-to-end pipeline used in our paper: data collection & formatting, graph construction, feature generation, machine-learning experiments (tabular and graph), statistical analysis, and figure generation.

> **Repro tip:** run commands from the `scripts/` directory so relative paths like `../data/...` and `../output/...` resolve correctly.

---

## Table of Contents

- [Repository Layout](#repository-layout)
- [Environment](#environment)
- [‼️ Data Access (Not Provided)](#️-data-access-not-provided)
- [End-to-End Workflow](#end-to-end-workflow)
  - [1) Collect Data](#1-collect-data)
  - [2) Format Data](#2-format-data)
  - [3) Machine Learning](#3-machine-learning)
  - [4) Evaluate & Plot](#4-evaluate--plot)
  - [5) Other Experiments](#5-other-experiments)
- [Script Details & Usage](#script-details--usage)
  - [Collect / Format](#collect--format)
  - [Graph Building & Features](#graph-building--features)
  - [GML Inputs](#gml-inputs)
  - [Tabular ML](#tabular-ml)
  - [Graph ML](#graph-ml)
  - [Baselines, Stats, Plots](#baselines-stats-plots)
- [Outputs](#outputs)
- [Notes on Paths](#notes-on-paths)
- [Citations & Acknowledgments](#citations--acknowledgments)
- [License](#license)
- [Quickstart](#quickstart-tldr)

---

## Repository Layout

```
repo/
├─ data/                       # not tracked; see Data Access below
├─ output/                     # generated artifacts (models, CSVs, figures)
├─ scripts/
│  ├─ collect_data/
│  │  ├─ XMLToCSV.py
│  │  └─ format_dblp.py
│  ├─ models/
│  │  ├─ GAT.py
│  │  ├─ GCN.py
│  │  ├─ GraphSAGE.py
│  │  ├─ RecurrentGCN.py      # temporal model (aka GConvGRU style)
│  │  └─ TabTransformer.py
│  ├─ build_graph.py
│  ├─ get_coauthorship_features.py
│  ├─ get_bibliometric_features.py
│  ├─ prep_for_gml.py
│  ├─ tabular_ml.py
│  ├─ graph_ml.py
│  ├─ explicit_baselines.py
│  ├─ stats.py
│  ├─ plot.py
│  └─ rewire.py
└─ README.md
```

---

## Environment

- Python ≥ 3.10
- Recommended: create a fresh virtual environment

Minimal dependencies (put these in `requirements.txt` if desired):

```
numpy
pandas
scikit-learn
matplotlib
seaborn
networkx
scipy
tqdm
lxml
torch             # pick version compatible with your CUDA/CPU
torch-geometric   # plus torch-scatter / torch-sparse per PyG install docs
joblib
```

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # or: pip install <packages above>
```

---

## ‼️ Data Access (Not Provided)

**We do not distribute raw or derived datasets in this repository.** To reproduce results you must obtain source data yourself:

1. **DBLP XML dump and DTD**  
   Download the official DBLP XML and its DTD (see DBLP documentation).  
   You will pass these to `XMLToCSV.py`.

2. **Faculty roster**  
   We used the computer-science faculty roster from Jeff Huang’s “computer-science-open-data”.  
   Save to: `data/hiring/faculty_raw.csv`

3. **University rankings**  
   We used CSRankings.  
   Save to: `data/university_ranking/csrankings.csv`

4. **Where files end up**  
   - `XMLToCSV.py` writes per-element CSVs (we assume `data/coauthorship/dblp_raw/`).  
   - `format_dblp.py` reads those CSVs plus the roster/rankings and writes cleaned tables:
     - `data/coauthorship/dblp_clean/`, `data/coauthorship/dblp_cleaned/`
     - updates `data/hiring/faculty.csv`

> If you cannot access these sources, you can run the modeling and plotting on your own similarly structured data. See each script’s “Expected inputs/outputs”.

---

## End-to-End Workflow

Run from `scripts/`:

### 1) Collect Data
1. **DBLP → CSVs:** `collect_data/XMLToCSV.py`  
   Parses DBLP XML with DTD and writes per-element CSVs.
2. **Clean & integrate:** `collect_data/format_dblp.py`  
   Merges DBLP, faculty roster, and rankings. Produces author-level and faculty tables.

### 2) Format Data
1. **Build graphs & splits:** `build_graph.py`  
   Creates author–paper incidence matrices, co-authorship adjacency matrices, and train/val/test splits.
2. **Features:**  
   `get_coauthorship_features.py` (graph features per year)  
   `get_bibliometric_features.py` (bibliometric features per year)
3. **Pack for GML:** `prep_for_gml.py`  
   Packs yearly adjacencies, builds feature tensors, labels for multiple thresholds, and boolean masks.

### 3) Machine Learning
- **Tabular models:** `tabular_ml.py` — grid search + TabTransformer; repeats for stability across top ∈ {10,20,30,40,50}.
- **Graph models:** `graph_ml.py` — static (GCN/GAT/GraphSAGE) and temporal (RecurrentGCN/GConvGRU-style) with sweeps, selection, repeats; parallelizable per config.

### 4) Evaluate & Plot
- **Baselines:** `explicit_baselines.py`  
- **Stats:** `stats.py` (LME & reference models)  
- **Figures:** `plot.py` (paper figures; includes rewiring figure via `rewire()`)

### 5) Other
- **Degree-preserving rewiring experiments:** `rewire.py`

---

## Script Details & Usage

> All commands assume: `cd scripts/`

(Details omitted for brevity — see conversation log for full text.)
