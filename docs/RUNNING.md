# Execution commands

Run commands from the repository root after installing dependencies. Paths under `inputs/` are examples for files supplied by the user; they are not bundled downloads. All generated files go under `work/` by default. Commands train models and may require substantial GPU time.

## Data preparation

Map candidate accessions to supplied UniRef50 clusters:

```bash
python scripts/map_uniref50.py --input inputs/raw_2026.tsv --mapping inputs/idmapping_2026_04_01.tsv --output inputs/uniref_2026.tsv
python scripts/prepare_ec_go_labels.py --input inputs/uniref_2026.tsv --go-obo inputs/go-basic.obo --slim-obo inputs/goslim_generic.obo --outdir inputs/labels
python scripts/prepare_localization_labels.py --input inputs/uniref_2025.tsv --output inputs/localization_multihot.csv
```

The 2025 source may also be passed to `map_uniref50.py` with its corresponding mapping. To rebuild a particular published input set, supply that set's original prepared candidate table and label order instead of substituting a newly mapped table.

Collect common accessions from all the relevant pregenerated representations. Pass every file used in the original availability filter, not just the two files shown as argument examples:

```bash
python scripts/collect_available_entries.py --embeddings inputs/encoder1_strategy1.npz inputs/encoder2_strategy1.npz --output inputs/available_entries.csv
python scripts/prepare_datasets.py --task EC_level2 --sequences inputs/labels/ec_level2_dataset.csv --domains inputs/ec_domains.json --available-entries inputs/ec_available_entries.csv
python scripts/prepare_datasets.py --task GO_slim --sequences inputs/labels/final_slim_dataset.csv --domains inputs/go_domains.json --available-entries inputs/go_available_entries.csv
python scripts/prepare_datasets.py --task Subcellular --sequences inputs/uniref_2025.tsv --labels inputs/localization_multihot.csv --domains inputs/localization_domains.json --available-entries inputs/localization_available_entries.csv
```

The default random split seed is 42. Reuse the resulting split files for every representation and predictor within a task.

## Full-length encoding and regional pooling

```bash
python scripts/embed_full_length_residues_esm2.py
python scripts/embed_full_length_residues_prott5.py
python scripts/pool_full_length_mean.py --model esm2
python scripts/pool_full_length_mean.py --model prott5
python scripts/pool_full_length_domain.py --model esm2
python scripts/pool_full_length_domain.py --model prott5
python scripts/pool_domain_random.py
python scripts/train_predictors.py --strategies full_length_mean full_domain_pooling domain_matched_random_region_only_seed101 domain_matched_random_region_only_seed102 domain_matched_random_region_only_seed103 domain_matched_random_region_only_seed104 domain_matched_random_region_only_seed105
```

These pooling strategies use the same full-length residue matrices. The random control matches the number of domain residues, with seeds 101–105. The downstream command uses the unified validation protocol described in the README.

## Approximately 1,000-residue inputs

```bash
python scripts/encode_positional_inputs.py --model esm2
python scripts/encode_positional_inputs.py --model prott5
python scripts/train_predictors.py --strategies full_length_mean head1000 mid1000 tail1000 splice300_400_300 domain_center_longest domain_max_cover
```

The six local strategies correspond to Table 2. The ProtT5 utility deliberately retains the `[1:-1]` fragment-pooling operation and the localization-specific longest-domain boundary rule described in Supplementary Methods S2. It must not be interpreted as a standardized all-residue ProtT5 comparison. Full-length and 300-residue encoding retain all residue vectors.

## Supervised 300-residue windows

```bash
python scripts/build_signal_fragment_coordinates.py
python scripts/encode_signal_fragments.py --embedding-model esm2
python scripts/encode_signal_fragments.py --embedding-model prott5
python scripts/evaluate_signal_fragment_results.py
```

Window scoring uses a training-only linear classifier with validation-selected regularization. The selected window is the earliest maximum of the rolling sum of residue RMS scores. The five random-window seed derivations follow the original implementation.

## Teacher and student 500-residue windows

```bash
python scripts/build_crossfit_window_teacher.py
python scripts/train_window_student.py
python scripts/encode_window_distillation_esm2.py
python scripts/evaluate_window_distillation_results.py
```

The teacher uses five-fold cross-fitting of its linear classifier within the development set. The student is trained using the original training sequences and selected by validation distillation loss; it is not refitted on the combined development set. This is distinct from final downstream predictor refitting.

Generate and evaluate the EC N-terminal control separately:

```bash
python scripts/build_ec_head500_manifest.py
python scripts/encode_window_distillation_esm2.py --tasks EC_level2 --strategies head500 --manifest work/analysis_outputs/ec_demo/head500_manifest.csv.gz
python scripts/evaluate_window_distillation_results.py --tasks EC_level2 --strategies head500
```

After training a student, select windows on new raw sequences without running a full-length PLM:

```bash
python scripts/select_student_windows.py --checkpoint work/analysis_outputs/window_distillation/student/EC_level2__student.pt --input inputs/new_sequences.csv --output work/selected_windows.csv
```

## Evaluation and checks

Training entry points write test predictions, selected settings, thresholds, and metrics. Recalculate a saved test Macro-F1 with:

```bash
python scripts/evaluate_predictions.py --predictions work/downstream_results/esm2/EC_level2/full_length_mean/lasso/predictions.npz
python -m unittest discover -s tests -v
```

The unit tests use synthetic data. They check data splitting, threshold/refitting behavior, window coordinates, random residue-count matching, and student inference. They do not rerun the manuscript's PLM encoding or full benchmarks.
