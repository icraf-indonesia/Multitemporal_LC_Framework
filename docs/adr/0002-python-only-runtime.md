# Python-only repository dropping GEE JavaScript Code Editor parity

status: accepted

`LC_Framework` is maintained strictly as a Python package (`earthengine-api` for GEE compute and scientific Python for local/Colab compute), dropping the dual GEE JavaScript (`src/`) + Python (`python/`) layout inherited from `Temporal_segment_gee`. The target team operates in Google Colaboratory and Python, where maintaining parallel JavaScript modules creates a high risk of silent scientific divergence and doubles testing overhead. All GEE workflows are authored once in Python and tested offline with `pytest`.
