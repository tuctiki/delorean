---
description: Ensure all python and pip commands run in the 'quant' conda environment
---

To ensure that you are always using the correct environment for this project, you must:

1.  **NEVER** run `python` or `pip` directly.
2.  **ALWAYS** prefix these commands with `conda run -n quant`.

Examples:
- **Incorrect**: `python script.py`
- **Correct**: `conda run -n quant python script.py`

- **Incorrect**: `pip install pandas`
- **Correct**: `conda run -n quant pip install pandas`

This applies to all `run_command` usages involving python execution.
