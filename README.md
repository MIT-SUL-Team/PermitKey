# PermitKey

Code for *A harmonized national work-type classification of U.S. building permit records*
(Martinez, Lewis, Fan and Zheng).

PermitKey classifies U.S. building-permit records into 14 work-type categories from their
free-text descriptions, using the jurisdictions' own permit labels as weak supervision, and
aggregates the classified records to counts by 2020 census tract and calendar year.

The released dataset, model checkpoint and validation tables are on Harvard Dataverse:
https://doi.org/10.7910/DVN/YXCXLQ

## Repository layout

| Folder | Contents |
| --- | --- |
| `src/permitkey_release/` | Installable package: model inference, the solar/EV generic-label rule, tract-year aggregation, human-review metrics |
| `acquisition/` | Downloaders for Socrata, ArcGIS and Carto open-data portals, with one example source each |
| `taxonomy/` | Native-label embedding, concept assignment and clustering used to build the taxonomy |
| `pipeline/` | Record normalization (descriptions, native types, dates, locations) and corpus-scale classification |
| `training/` | Weak-label construction, model training and city-held-out evaluation |
| `config/` | Released categories and definitions, training concepts, native-label crosswalk, generic-label rule |
| `validation/` | Human-review tasks, reviewer votes, per-category metrics and the city-held-out benchmark |
| `scripts/` | Rebuild the released table, reproduce the human-review metrics, check model parity |
| `examples/`, `tests/` | Synthetic inputs and unit tests |

## Install

```
python -m pip install .
```

## Usage

Classify permit records (JSON lines with `uid`, `description`, `native_type`, `native_subtype`)
with the released checkpoint, `permitkey_model.joblib` from Dataverse:

```
permitkey classify --input examples/classifier_inputs.jsonl --output predictions.jsonl --model permitkey_model.joblib
```

Rebuild the tract-year table from its source-contribution file on Dataverse:

```
python scripts/rebuild_release.py --input permitkey_tract_year_sources.parquet --output rebuilt
```

Reproduce the human-review micro-F1 (79.9%) and macro-F1 (79.2%):

```
python scripts/check_review.py validation/human_completed_tasks.json
```

Check that the checkpoint reproduces the saved scores of the reviewed permits:

```
python scripts/check_runtime_parity.py permitkey_model.joblib
```

Run the tests:

```
python -m unittest discover -s tests
```

Scripts in `acquisition/`, `pipeline/` and `training/` read inputs from and write outputs to a
local `data/` directory. Raw permit records come from the public portals listed in the source
manifest on Dataverse (`permitkey_sources.csv`).

## License

Code: MIT (see `LICENSE`). Dataset and model checkpoint: CC BY 4.0 (Harvard Dataverse).

## Citation

Martinez, C., Lewis, S. H., Fan, Y. and Zheng, S. A harmonized national work-type classification
of U.S. building permit records. Harvard Dataverse, https://doi.org/10.7910/DVN/YXCXLQ
