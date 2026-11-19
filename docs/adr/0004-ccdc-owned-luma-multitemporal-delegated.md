# Own CCDC temporal segmentation natively; delegate direct multitemporal orchestration to luma-stack

status: accepted

`LC_Framework` natively implements Continuous Change Detection and Classification (CCDC) temporal segmentation, synthetic reflectance reconstruction, harmonic extraction, and temporal stability masking in `lc_framework.temporal`. Direct multi-period GEE compositing and multitemporal model transfer remain delegated to `luma-stack` (via `lc_framework.luma_adapter`) as a secondary option, avoiding duplication of `luma-stack`'s active `multitemporal_analysis_support` work while filling its missing CCDC capability.
