# experiments/tools/

Maintenance scripts and the interactive mask viewer.

| File | Purpose |
|---|---|
| `verify_gap_table.py` | Checks a stored gap table against the current mask and DEM by regenerating its first chunk, then stamps it with the current fingerprint. Run after changing a mask. |
| `audit_results.py` | One-time migration: stamps estimator result files written before input fingerprints existed, when they are newer than every input they depend on. |
| `partial_mask_viewer.ipynb` | Notebook to build a mask step by step (base, augment, closure, prune), report its size and crossing mass, and draw it on the code patch. |

The notebook finds the repository root from its working directory, so it
can be opened from any folder inside the checkout.
