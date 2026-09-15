# Required inputs

The initial repository upload contains scripts only. Supply the following inputs locally before running the corresponding stages.

## Raw sequence sources

| Task | Source export |
| --- | --- |
| Subcellular localization | `uniprotkb_length_100_TO_AND_annotation_2025_08_14.tsv` |
| EC and GO slim | `uniprotkb_length_1201_TO_AND_reviewed_t_2026_03_31.tsv` |

These filenames identify exports, not official UniProt database release numbers. The exports originate from UniProtKB, https://www.uniprot.org/. They are not downloadable from this repository in the initial code upload.

UniRef50 mapping requires a two-column UniProt ID-mapping TSV. GO construction requires the `go-basic.obo` and `goslim_generic.obo` resources corresponding to 2026-01-23. Domain caches are JSON objects mapping each accession to a list of one-based inclusive `[start, end]` pairs. These auxiliary inputs are not bundled. Later database downloads may produce different annotations or sequences.

## Constructing final analysis inputs

`prepare_datasets.py` requires the candidate sequence/label table, domain cache, and a frozen common-representation accession inventory. `collect_available_entries.py` constructs such an inventory from supplied embedding archives. The manuscript's analysis-set construction used availability across eight pregenerated inputs under both encoders; the six local strategies in Table 2 are a subset of that preprocessing inventory.

The preparation command sorts accessions before applying the original random 80/10/10 split with seed 42. Exact reproduction of the manuscript's split membership requires the same candidate rows, annotation versions, common-representation inventory, and label-column order. Supplying only a current database download does not guarantee the same analysis set. Keep the original localization multi-hot column order when supplying `--labels`; freshly constructing labels may order those columns differently.

The resulting directory layout is:

```text
work/
  data/
    EC_level2_master.csv
    GO_slim_master.csv
    Subcellular_master.csv
    EC_level2_domains.json
    GO_slim_domains.json
    Subcellular_domains.json
    all_unique_full_length_entries.csv
  labels/
    <task>_labels_multihot.csv
    <task>_label_list.csv
  splits/
    <task>_split_entries.npz
```

Each master CSV contains `Entry`, `Length`, `Sequence`, and `DomainCount`. Each split NPZ contains `train_entries`, `val_entries`, `test_entries`, `y_train`, `y_val`, `y_test`, and `label_cols`. The label arrays are binary two-dimensional matrices. Entry order is meaningful and is used to align representations.

The manuscript's final analysis sizes are:

| Task | Training | Validation | Test | Labels |
| --- | ---: | ---: | ---: | ---: |
| EC_level2 | 1500 | 187 | 188 | 10 |
| GO_slim | 3371 | 421 | 422 | 43 |
| Subcellular | 3773 | 472 | 472 | 21 |

The original shared full-length sequence table must be retained when reproducing an existing encoding run, particularly when the same accession has different sequence versions in different exports. A newly constructed union is suitable for a new run and is not a substitute for that archived table.

## Representation files

Full-length residue files are `work/residue_embeddings/<encoder>/all_unique/<Entry>.npz`, each with a `residues` matrix of shape `(sequence_length, embedding_dimension)`. Pooled NPZ archives map accession keys to vectors. ESM-2 vectors have 1,280 features; ProtT5 vectors have 1,024.

All fragment coordinates are zero-based and half-open. Fragment manifests include SHA-256 hashes checked before encoding. The 300-residue and 500-residue encoders are run on the extracted amino acid strings independently of full-length encoding.
