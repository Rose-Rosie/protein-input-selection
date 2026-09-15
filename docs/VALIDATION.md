# Code validation

Nine synthetic workflow tests passed for this code upload. The tests exercise all three downstream predictors through the full-length, 300-residue, and student-window evaluation entry points; check that thresholds are selected before combined-data refitting; verify the final scaler and fixed DNN epochs; and check domain-random residue counts, positional boundary behavior, teacher/student distributions, dataset splitting, and saved-prediction Macro-F1.

The core verification environment used NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1, scikit-learn 1.6.1, PyTorch 2.5.1+cu121, and XGBoost 2.1.4. The requirements file gives installation ranges rather than claiming to reconstruct the original training environment.

Pretrained PLM inference, original ontology processing, and full manuscript benchmarks were not rerun for this upload. Synthetic test data are generated in temporary directories and are not study data.
