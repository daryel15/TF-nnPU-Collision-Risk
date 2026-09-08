# Version 0.1.0 packaging changes

- Portable notebook entry points and root-relative input paths; default execution never starts long training.
- Command-line evaluation with metric assertions and keyed prediction exports.
- Strict manifest/data/split/mask checks, safe checkpoint loading, protected release input directories, and separate timestamped outputs.
- Corrected common-current-risk P-to-U masks to the seed schedule actually present in the GAT/CMPA source; kept original split indices and result tables.
- Extracted baseline model implementations into readable Python runners; full runs export weights, settings, scalers or Kalman parameters, and predictions.
- Added isolated seeded Stage-2, sensitivity and cost-sensitive reruns. Their provenance limitations are explicit.
- Recovered complete symmetric/P-to-U F1 tables and other available tables from saved notebook HTML output.
- Historical source preserved as Markdown; pasted invalid table rows are comments and are not part of runnable notebooks.
- Added installation/kernel commands, direct dependency pins, optional profiling requirements, data dictionary, citation/license files, method notes, and validation instructions.
- Clarified the historical negative-risk loss without changing it or replacing the supplied model weights.
