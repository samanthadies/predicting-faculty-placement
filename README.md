# Forecasting Faculty Placement from Patterns in Co-authorship Networks

This repository contains the end-to-end pipeline used in our paper: data collection & formatting, graph construction, feature generation, machine-learning experiments (tabular and graph), statistical analysis, and figure generation.

If you found this code useful for your own research, please cite our paper.

```bibtex
@article{dies2025forecasting,
  title={Forecasting Faculty Placement from Patterns in Co-authorship Networks},
  author={Dies, Samantha and Liu, David and Eliassi-Rad, Tina},
  journal={arXiv preprint arXiv:2507.14696},
  year={2025}
}
```
---

## Table of Contents

- [Repository Layout](#repository-layout)
- [Data Access (Not Provided)](#️-data-access-not-provided)
- [End-to-End Workflow](#end-to-end-workflow)
  - [1) Collect Data](#1-collect-data)
  - [2) Format Data](#2-format-data)
  - [3) Machine Learning](#3-machine-learning)
  - [4) Evaluate & Plot](#4-evaluate--plot)
  - [5) Other Experiments](#5-other-experiments)
- [Script Details & Usage](#script-details--usage)
- [Outputs](#outputs)
- [Citations & Acknowledgments](#citations--acknowledgments)

---

## Repository Layout

```
repo/
├─ data/                       # not tracked; see Data Access below
├─ output/                     # generated artifacts (models, CSVs, figures)
├─ scripts/
│  ├─ collect_data/
│  │  ├─ XMLToCSV.py
│  │  ├─ scrape.py
│  │  ├─ infer_gender.py
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

## Data (Not Provided)

**We do not distribute raw or derived datasets in this repository, but all data is open source.** To reproduce results you must obtain source data yourself:

1. **DBLP [1]**  
   Download the official DBLP XML and its DTD from [https://dblp.org/xml/release/dblp-2023-1-03.xml.gz][https://dblp.org/xml/release/dblp-2023-1-03.xml.gz].
   Save to: `data/coauthorship/dblp_raw/dblp.dtd` and `data/coauthorship/dblp_raw/dblp.xml`

2. **Faculty Data [2]**  
   Use `scrape.py` to collect the faculty dataset from Jeff Huang's CS Professors page ([https://drafty.cs.brown.edu/csprofessors]).
   Saves to: `data/hiring/faculty.py`

4. **University rankings [3]**  
   Download department ranks from CSRankings ([https://csrankings.org/][https://csrankings.org/]).
   Save to: `data/university_ranking/csrankings.csv`

5. **Where files end up**  
   - `XMLToCSV.py` writes per-element CSVs to `data/coauthorship/dblp_raw/`.  
   - `format_dblp.py` reads those CSVs, plus the faculty data and rankings, and writes cleaned tables:
     - `data/coauthorship/dblp_clean/full_by_author.csv`, `data/coauthorship/dblp_cleaned/full_by_paper.csv`
     - updates `data/hiring/faculty.csv`
---

## End-to-End Workflow

Run from `scripts/`:

### 1) Collect Data
- **Scrape faculty data** `collect_data/scrape.py`
   Scrapes Jeff Huang's CS Professors data to save faculty data.
- **DBLP to CSVs:** `collect_data/XMLToCSV.py`  
   Parses DBLP XML with DTD and writes per-element CSVs
- **Clean & integrate:** `collect_data/format_dblp.py`  
   Merges DBLP, faculty data, and rankings. Produces author-level and faculty tables
- **Gender label** `collect_data/infer_gender.py`
   Infers gender using labels from SOTA proprietary APIs, Gender-API [4] and genderize.io [5].
   If Gender-API and genderize.io both have confidence >= 0.8 and agree, we use that gender label.
   Otherwise, we manually label the gender.

### 2) Format Data
- **Build graphs & splits:** `build_graph.py`  
   Creates author–paper incidence matrices, co-authorship adjacency matrices, and train/val/test splits
- **Features:**  
   `get_coauthorship_features.py` (graph features per year)  
   `get_bibliometric_features.py` (bibliometric features per year)
- **Prepare for GML:** `prep_for_gml.py`  
   Packs yearly adjacencies, builds feature tensors, labels for multiple thresholds, and boolean masks

### 3) Machine Learning
- **Tabular models:** `tabular_ml.py`
   Hyperparameter search and final runs for tabular models
   Repeats for stability across top in {10,20,30,40,50}
- **Graph models:** `graph_ml.py`
   Hyperparameter search and final runs for graph models
   Repeats for stability across top in {10,20,30,40,50}

### 4) Evaluate & Plot
- **Baselines:** `explicit_baselines.py`  
- **Stats:** `stats.py`
- **Figures:** `plot.py`

### 5) Other
- **Degree-preserving rewiring experiments:** `rewire.py`

---

## Script Details & Usage

Each script can be run independently once its inputs exist. Example commands:

#### Scrape faculty data
```
python collect_data/scrape.py
```

#### Collect & format DBLP
```
python collect_data/XMLToCSV.py dblp.xml dblp.dtd ../data/coauthorship/dblp_raw/ --annotate
python collect_data/format_dblp.py
```

#### Build graphs & features
```
python build_graph.py
python get_coauthorship_features.py
python get_bibliometric_features.py
python prep_for_gml.py
```

#### Tabular models
`python tabular_ml.py`

#### Graph models (example: GCN, bibliometric features, top=20)
This script is parallelized so that each iteration trains a single model based on command-line inputs.
```
python graph_ml.py hyperparam_sweep GCN biblio 20 0
python graph_ml.py best_hyperparams GCN biblio 20
python graph_ml.py repeat_best GCN biblio 20 10
python graph_ml.py analyze_repeated_runs 20
```

#### Evaluation
```
python explicit_baselines.py
python stats.py
python plot.py
```

---

## Outputs

All outputs are written to the `output/` directory.
Representative structure:

```
output/
├─ y_*/
│  ├─ experiment_summary.csv
│  ├─ repeat_results.csv
│  ├─ best_simple_models/
│  ├─ gml_combined_runs.csv
│  ├─ gml_aggregates.csv
│  ├─ gml_aggregates_wide.csv
│  ├─ gml_aggregates_ranked_by_pr_auc.csv
│  └─ gml_aggregates_ranked_by_mcc.csv
├─ models_biblio/
│  └─ GCN/
│     └─ y_20/
│        ├─ sweep/
│        ├─ best_configs.csv
│        ├─ repeat/
│        └─ analysis/
├─ stats/
│  ├─ mixed_effects_*.txt
├─ plots/
│  ├─ barplot_y_*.pdf
│  ├─ boxplot_y_*.pdf
│  ├─ rewired.pdf
│  └─ lineplot_summary.pdf
└─ rewired/
   └─ rewired_no_graph_feats_summary.csv
```

---

## Citations & Acknowledgements

[1] DBLP Computer Science Bibliography. [https://dblp.org/xml/release/dblp-2023-1-03.xml.gz](https://dblp.org/xml/release/dblp-2023-1-03.xml.gz) (2023).  

[2] Huang, J. Computer Science Open Data. [https://jeffhuang.com/computer-science-open-data/](https://jeffhuang.com/computer-science-open-data/) (2022).  

[3] Berger, E. D. CSRankings. [https://csrankings.org/](https://csrankings.org/) (2023).

[4] [https://gender-api.com](https://gender-api.com).

[5] [https://genderize.io](https://genderize.io).

If you use this codebase or reproduce its analyses, please cite:
> Dies, Samantha, David Liu, and Tina Eliassi-Rad. *Forecasting Faculty Placement from Patterns in Co-authorship Networks*.
arXiv preprint arXiv:2507.14696 (2025).
