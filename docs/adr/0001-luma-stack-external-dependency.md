# Treat luma-stack as an external dependency via adapters

status: accepted

We keep `luma-stack` (`luma_ge`) as an external `pip`-installed dependency integrated through a thin adapter module (`lc_framework.luma_adapter`), rather than copying or rewriting its source code into `LC_Framework`. The other three legacy repositories (`Temporal_segment_gee`, `Improve_pixel_based_lc_classification`, and `similarity_search_embeddings`) live in a personal GitHub account and are being consolidated and rewritten natively into this company repository, whereas `luma-stack` has its own active lifecycle and team maintenance. Integrating via a dedicated adapter avoids code duplication and version drift while keeping `LC_Framework`'s core modules decoupled from `luma_ge` internals.
