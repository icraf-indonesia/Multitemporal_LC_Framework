# Keep every workflow subpackage standalone with lazy optional imports

status: accepted

Each stage subpackage in `lc_framework` (`temporal`, `features`, `samples`, `classification`, `obia`, `assembly`, `evaluation`) must be usable independently without installing or initializing upstream stages. Top-level package imports must never eagerly import optional dependencies (`earthengine-api`, `luma_ge`, `lightgbm`, `xgboost`, `scikit-image`), and downstream functions (such as `lc_framework.assembly` and `lc_framework.evaluation`) must accept direct file paths or rasters in addition to `ProjectConfig`, so analysts can run any single stage (e.g., Classification Assembly on externally generated GeoTIFFs) with only the lean core install.
