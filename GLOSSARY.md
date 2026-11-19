# LC Framework

A unified land cover and land use (LULC) mapping framework spanning pixel-based, geo-object-based, and hybrid pixel-object probability classification across Google Earth Engine and Python/Colab.

## Language

### Core Data Contracts

**Class Schema**:
An ordered, project-specific catalog of land cover classes defining each class's unique integer identifier, canonical name, and visualization color.
_Avoid_: Legend, class dictionary, label map

**Class Probability Stack**:
A multi-band spatial dataset aligned to a Class Schema where each band holds the per-pixel probability of a class, summing to 1.0 across valid pixels.
_Avoid_: Soft map, probability raster, confidence image

**Segment Label Raster**:
A single-band spatial raster assigning each pixel to a unique integer geo-object identifier, aligned to the same grid as a Class Probability Stack.
_Avoid_: Superpixel map, cluster image, object mask

### Classification Paradigms

**Pixel-Based Classification**:
A classification paradigm that estimates a Class Probability Stack independently for each pixel from spectral, temporal, radar, terrain, or embedding features.
_Avoid_: Per-pixel model

**Geo-Object Classification**:
A classification paradigm that partitions imagery into geo-objects, aggregates features per object, classifies each object probabilistically, and projects the resulting probabilities onto a Class Probability Stack.
_Avoid_: OBIA (when referring specifically to the classification paradigm rather than segmentation), object-based mapping

**Object-Refined Pixel Probabilities**:
A hybrid classification strategy that combines pixel-level class probabilities with geo-object mean probabilities through adaptive blending per segment.
_Avoid_: Hybrid classification (when used ambiguously without specifying the blending strategy)

### Workflow Stages

**Temporal Segment**:
A continuous time interval fitted by harmonic regression (CCDC) between spectral breakpoints, used to reconstruct cloud-free synthetic reflectance and assess pixel stability.
_Avoid_: CCDC slice, time window

**Sample Augmentation**:
The expansion of labeled training samples by identifying unlabeled locations with high similarity to existing class prototypes in a satellite embedding space.
_Avoid_: Pseudo-labeling (unless explicitly trained iteratively), similarity classification

**Classification Assembly**:
The downstream stage that consumes pixel-level class probabilities (and optional geo-objects or ancillary masks) to produce refined probability maps, hard land cover maps, uncertainty layers, and area or trajectory statistics.
_Avoid_: Post-processing (when referring to the full probability fusion and refinement stage), assembly script

**Confusion Index**:
A per-pixel or per-object classification uncertainty metric defined as the ratio of the second-highest class probability to the highest class probability ($S / W$), used to weight hybrid probability blending and flag ambiguous regions.
_Avoid_: Uncertainty score (when referring specifically to the $S / W$ ratio)
