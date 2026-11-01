# Data scope

This release distributes source code and an independent synthetic generator. No original field rows, well identifiers, production measurements, train/test extracts, trained checkpoints, normalization statistics, or saved field figures are included.

## Included

- Reusable Python functions.
- A complete synthetic generation procedure with a fixed seed and generic categories.
- An output-free notebook that calls that procedure.
- Dependency lists and setup documentation.

The synthetic values are invented from simple formulas. They are not anonymized, perturbed, or sampled versions of the original field records. Generated output is labeled synthetic and belongs in the ignored `artifacts/` directory or temporary storage.

## Excluded

Raw and processed project tables; legacy field-analysis notebooks and their outputs; trained weights; data-derived normalization parameters and run metadata; original surface CSV, PNG, and HTML files; and saved field diagnostics are excluded from the release tree. The original data's redistribution terms have not been established, so those materials are not supplied here.

Feature names in source describe expected input schema, not original measurements. A schema name does not grant rights to any external dataset. Use only data you are authorized to process and share.

No new software or data license is introduced by this release. Existing source notices are preserved. Dependencies remain subject to their own licenses.
