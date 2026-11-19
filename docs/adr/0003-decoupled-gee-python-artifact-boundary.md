# Decouple GEE ingestion/classification from local assembly via artifact contracts

status: accepted

Heavy imagery ingestion, temporal segmentation, and default pixel classification execute in Google Earth Engine, while Classification Assembly, optional local ML backends, and evaluation execute in Python/Colab, communicating strictly through standardized hand-off artifacts (`Class Probability Stack` and `Segment Label Raster`). Rather than tightly coupling both runtimes through live streaming, artifact-decoupled stages let analysts swap upstream classifiers (GEE RF, `luma-stack`, local LightGBM/XGBoost, or future RSFM models) and swap segmentation providers (GEE SNIC vs. local SLIC) independently without rewriting downstream assembly or evaluation pipelines.
