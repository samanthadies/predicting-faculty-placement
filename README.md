# predicting-faculty-placement


## DBLP XML to CSV Converter

Convert the official **DBLP XML dump** (validated against its DTD) into tidy CSVs — one file per DBLP element (e.g., `article`, `inproceedings`, `book`).  
Optionally include annotated headers, Neo4j-friendly imports, and relationship CSVs (e.g., for authors).

---

### Features

- DTD-validated parsing with **lxml**
- Automatic schema discovery (per-element attributes & child tags)
- One CSV per DBLP element, each with a synthetic integer `id` column
- Optional annotated headers with inferred column types
- Optional relation CSVs for selected attributes (e.g., authors)
- Optional `neo4j-admin import` command generation

---

### How to run

Run the script with the DBLP XML file, its DTD, and a base output filename:

`python dblp_to_csv.py dblp.xml dblp.dtd out.csv --relations author:authored_by --annotate --neo4j`
