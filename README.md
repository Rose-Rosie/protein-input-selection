# Protein input selection

Code for **Local sequence information and task-dependent fragment selection for protein function prediction**.

The repository covers three multi-label tasks: EC subclass prediction (`EC_level2`), GO slim term prediction (`GO_slim`), and subcellular localization (`Subcellular`). It provides data-processing scripts, ESM-2 and ProtT5 encoding, regional pooling, sequence-window selection, and training and evaluation of L1-regularized logistic regression, XGBoost, and a downstream DNN.

This initial upload contains code and documentation. Raw data, prepared datasets, embeddings, trained weights, and prediction results are not included.

## Installation

Use Python 3.10 or 3.11 in a separate environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

PLM encoding requires a suitable CUDA GPU and downloads pretrained weights from their original providers. Install the PyTorch build appropriate for your CUDA installation. Dependency ranges describe the supported setup; they are not a lock file for the original training environment.

## Code corresponding to the manuscript

| Manuscript component | Main scripts |
| --- | --- |
| Data preprocessing and splits | `map_uniref50.py`, `prepare_ec_go_labels.py`, `prepare_localization_labels.py`, `collect_available_entries.py`, `prepare_datasets.py` |
| Full-length residue representations | `embed_full_length_residues_esm2.py`, `embed_full_length_residues_prott5.py` |
| Full-length, domain, and matched-random pooling | `pool_full_length_mean.py`, `pool_full_length_domain.py`, `pool_domain_random.py` |
| Approximately 1,000-residue positional and domain inputs | `encode_positional_inputs.py` and its two encoder utility modules |
| Supervised 300-residue selection | `build_signal_fragment_coordinates.py`, `encode_signal_fragments.py` |
| Teacher distributions and student selection | `build_crossfit_window_teacher.py`, `train_window_student.py`, `select_student_windows.py`, `encode_window_distillation_esm2.py` |
| EC N-terminal 500-residue control | `build_ec_head500_manifest.py` |
| Three downstream predictors | `train_predictors.py`, `evaluate_signal_fragment_results.py`, `evaluate_window_distillation_results.py` |
| Evaluation of saved predictions | `evaluate_predictions.py` |

All scripts are under `scripts/`. `lasso` is the retained code identifier for **L1-regularized logistic regression**, not linear regression. The student CNN and downstream DNN are separate models.

See [input requirements](docs/INPUTS.md) and [execution commands](docs/RUNNING.md). Scripts resolve generated files under `work/`; set `PROTEIN_INPUT_WORKDIR` to use another directory.

## Validation protocol and release status

Runnable downstream entry points implement the unified protocol in the main-methods draft: fit candidates and preprocessing on training data, use validation Macro-F1 to choose hyperparameters, select each label's decision threshold on validation predictions, fix thresholds and DNN epochs, and refit on training plus validation data before test evaluation. The threshold grid is 0.10–0.90 in steps of 0.05, with ties resolved toward the lower threshold.

The retained regional-pooling scores in the manuscript have not been recomputed using this unified protocol. This code upload is therefore an implementation release, not certification that rerunning it reproduces those scores. Full benchmark reproduction also requires the original prepared inputs and environment. The approximately 1,000-residue ProtT5 implementation preserves the task-specific boundary and token-pooling behavior documented in Supplementary Methods S2.

## License

Project code is provided under the [MIT License](LICENSE). External protein databases, ontology files, pretrained weights, and software dependencies retain their respective licenses. Raw UniProt exports are not part of this upload.
