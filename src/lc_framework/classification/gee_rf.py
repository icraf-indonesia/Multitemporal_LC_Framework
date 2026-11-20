"""Google Earth Engine Pixel-Based Classification via MULTIPROBABILITY Random Forest.

Consumes RFECV-selected features (or an explicit feature list), trains a GEE
``smileRandomForest(...).setOutputMode('MULTIPROBABILITY')`` classifier, emits
canonical ``prob_<id>`` bands ordered by ``ClassSchema``, and schedules
EECU-optimized ``uint16`` ``[0, 10000]`` tiled GeoTIFF / Asset exports paired
with validated sidecar JSON manifests (ADR-0003, ADR-0005).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lc_framework._gee import require_ee
from lc_framework.classification._common import resolve_class_schema, resolve_selected_features
from lc_framework.config import ClassSchema, ProjectConfig
from lc_framework.io import (
    UINT16_PROB_NODATA,
    UINT16_PROB_SCALE,
    ExportTilePlan,
    ProbabilityEncoding,
    ProbabilityManifest,
    generate_tile_grid,
    manifest_path_for,
    plan_tiled_gee_exports,
    submit_gee_export_tasks,
)


@dataclass(frozen=True)
class GEERFParams:
    """Hyperparameters for GEE ``ee.Classifier.smileRandomForest``."""

    number_of_trees: int = 200
    variables_per_split: int | None = None
    min_leaf_population: int = 1
    bag_fraction: float = 0.5
    max_nodes: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.number_of_trees <= 0:
            raise ValueError(f"number_of_trees must be > 0, got {self.number_of_trees!r}.")
        if self.variables_per_split is not None and self.variables_per_split <= 0:
            raise ValueError(
                f"variables_per_split must be > 0 when set, got {self.variables_per_split!r}."
            )
        if self.min_leaf_population <= 0:
            raise ValueError(
                f"min_leaf_population must be > 0, got {self.min_leaf_population!r}."
            )
        if not (0.0 < self.bag_fraction <= 1.0):
            raise ValueError(f"bag_fraction must lie in (0, 1], got {self.bag_fraction!r}.")
        if self.max_nodes is not None and self.max_nodes < 2:
            raise ValueError(f"max_nodes must be >= 2 when set, got {self.max_nodes!r}.")

    def to_ee_dict(self) -> dict[str, Any]:
        """Convert to exact Earth Engine ``smileRandomForest`` constructor kwargs."""
        payload: dict[str, Any] = {
            "numberOfTrees": int(self.number_of_trees),
            "minLeafPopulation": int(self.min_leaf_population),
            "bagFraction": float(self.bag_fraction),
            "seed": int(self.seed),
        }
        if self.variables_per_split is not None:
            payload["variablesPerSplit"] = int(self.variables_per_split)
        if self.max_nodes is not None:
            payload["maxNodes"] = int(self.max_nodes)
        return payload


@dataclass(frozen=True)
class GEERFModel:
    """Trained GEE ``MULTIPROBABILITY`` Random Forest bound to feature names and ClassSchema."""

    classifier: Any
    feature_names: tuple[str, ...]
    class_schema: ClassSchema
    class_property: str
    trained_class_ids: tuple[int, ...]
    params: GEERFParams


@dataclass(frozen=True)
class GEEProbabilityExportResult:
    """Planned/submitted GEE export tasks together with validated sidecar manifests."""

    plans: list[ExportTilePlan]
    tasks: list[Any]
    manifests: dict[str, ProbabilityManifest]
    manifest_paths: dict[str, Path]


def train_gee_rf(
    training_fc: Any,
    *,
    class_schema: ClassSchema | None = None,
    config: ProjectConfig | None = None,
    selected_features: Any | None = None,
    feature_image: Any | None = None,
    class_property: str = "class_id",
    params: GEERFParams | None = None,
    trained_class_ids: Sequence[int] | None = None,
    filter_to_schema: bool = True,
) -> GEERFModel:
    """Train a GEE ``MULTIPROBABILITY`` Random Forest on an RFECV or explicit feature subset."""
    schema = resolve_class_schema(class_schema, config, caller="train_gee_rf")
    ee = require_ee("GEE Pixel-Based Classification")
    rf_params = params if params is not None else GEERFParams()

    resolved_features = resolve_selected_features(selected_features)
    if resolved_features is None:
        if feature_image is None:
            raise ValueError(
                "Provide selected_features (list, SelectedFeaturesArtifact, or path) "
                "or feature_image to infer input bands."
            )
        input_properties: Any = feature_image.bandNames()
        stored_names: tuple[str, ...] = ()
        null_check_props: Any = input_properties.add(class_property)
    else:
        stored_names = resolved_features
        input_properties = list(resolved_features)
        null_check_props = [*input_properties, class_property]

    if trained_class_ids is None:
        # Assume training_fc contains samples for every schema class to avoid synchronous
        # .getInfo() RPCs during graph construction.
        resolved_trained_ids = schema.class_ids
    else:
        if not filter_to_schema:
            raise ValueError(
                "filter_to_schema cannot be False when trained_class_ids is explicitly set; "
                "out-of-schema classes in training_fc would misalign MULTIPROBABILITY arrays."
            )
        cleaned_ids = tuple(sorted({int(cid) for cid in trained_class_ids}))
        unknown = set(cleaned_ids) - set(schema.class_ids)
        if unknown:
            raise ValueError(
                f"trained_class_ids {sorted(unknown)} are not defined in ClassSchema "
                f"{schema.class_ids}."
            )
        if not cleaned_ids:
            raise ValueError("trained_class_ids cannot be empty.")
        resolved_trained_ids = cleaned_ids

    clean_fc = training_fc.filter(ee.Filter.notNull(null_check_props))
    if filter_to_schema:
        clean_fc = clean_fc.filter(ee.Filter.inList(class_property, list(resolved_trained_ids)))

    base_clf = ee.Classifier.smileRandomForest(**rf_params.to_ee_dict()).setOutputMode(
        "MULTIPROBABILITY"
    )
    trained = base_clf.train(
        features=clean_fc,
        classProperty=class_property,
        inputProperties=input_properties,
    )

    return GEERFModel(
        classifier=trained,
        feature_names=stored_names,
        class_schema=schema,
        class_property=class_property,
        trained_class_ids=resolved_trained_ids,
        params=rf_params,
    )


def scale_probability_image(
    prob_image: Any,
    *,
    encoding: ProbabilityEncoding = "uint16",
) -> Any:
    """Scale ``[0, 1]`` GEE probability bands to EECU-optimized ``uint16`` or ``float32``."""
    if encoding not in ("uint16", "float32"):
        raise ValueError(f"encoding must be 'uint16' or 'float32', got {encoding!r}.")
    if encoding == "uint16":
        return (
            prob_image.multiply(UINT16_PROB_SCALE)
            .round()
            .clamp(0.0, UINT16_PROB_SCALE)
            .toUint16()
        )
    return prob_image.toFloat()


def predict_gee_probabilities(
    model: GEERFModel,
    feature_image: Any,
    *,
    scale_output: bool = False,
    encoding: ProbabilityEncoding = "uint16",
) -> Any:
    """Classify ``feature_image`` into canonical ``prob_<id>`` bands ordered by ClassSchema."""
    require_ee("GEE probability prediction")
    selected_img = (
        feature_image.select(list(model.feature_names))
        if model.feature_names
        else feature_image
    )
    probs_array_img = selected_img.classify(model.classifier)

    trained_band_names = [f"prob_{cid}" for cid in model.trained_class_ids]
    probs_img = probs_array_img.rename("probs").arrayFlatten([trained_band_names])

    missing_ids = [
        cid for cid in model.class_schema.class_ids if cid not in model.trained_class_ids
    ]
    if missing_ids:
        ee = require_ee("GEE absent-class probability padding")
        zero_bands = [
            ee.Image.constant(0).float().rename(f"prob_{cid}") for cid in missing_ids
        ]
        for zb in zero_bands:
            probs_img = probs_img.addBands(zb)

    ordered = probs_img.select(list(model.class_schema.gee_prob_band_names))
    if scale_output:
        return scale_probability_image(ordered, encoding=encoding)
    return ordered


def get_gee_feature_importance(model: GEERFModel) -> Any:
    """Return server-side GEE ``ee.Dictionary`` of tree-split variable importances."""
    require_ee("GEE feature importance")
    explanation = model.classifier.explain()
    return explanation.get("importance")


def evaluate_gee_probabilities(
    model: GEERFModel,
    test_fc: Any,
    *,
    eps: float = 1e-7,
) -> dict[str, Any]:
    """Compute overall and per-class log loss and top-1 accuracy in Earth Engine."""
    if not (0.0 < eps < 0.1):
        raise ValueError(f"eps must lie in (0, 0.1), got {eps!r}.")
    ee = require_ee("GEE probability evaluation")

    ee_trained_ids = ee.List(list(model.trained_class_ids))
    ee_schema_ids = ee.List(list(model.class_schema.class_ids))
    ee_schema_names = ee.List(list(model.class_schema.class_names))
    scored = test_fc.classify(model.classifier, "probs")

    def _score_sample(feat: Any) -> Any:
        probs_arr = ee.Array(feat.get("probs"))
        probs_list = probs_arr.toList()
        true_id = ee.Number(feat.get(model.class_property))
        true_idx = ee_trained_ids.indexOf(true_id)
        # Guard indexOf == -1 so absent/untrained schema classes do not index from the end
        p_true = ee.Number(
            ee.Algorithms.If(
                true_idx.gte(0),
                ee.Number(probs_list.get(true_idx)).max(eps),
                ee.Number(eps),
            )
        )
        sample_loss = p_true.log().multiply(-1)
        pred_idx = ee.Number(probs_arr.argmax().get(0))
        is_correct = ee.Algorithms.If(true_idx.gte(0), pred_idx.eq(true_idx), 0)
        return feat.set({"_log_loss": sample_loss, "_correct": ee.Number(is_correct)})

    with_metrics = scored.map(_score_sample)
    per_class_loss = ee.Dictionary.fromLists(
        ee_schema_names,
        ee_schema_ids.map(
            lambda cid: ee.Algorithms.If(
                with_metrics.filter(ee.Filter.eq(model.class_property, cid)).size().gt(0),
                with_metrics.filter(ee.Filter.eq(model.class_property, cid)).aggregate_mean(
                    "_log_loss"
                ),
                None,
            )
        ),
    )
    per_class_acc = ee.Dictionary.fromLists(
        ee_schema_names,
        ee_schema_ids.map(
            lambda cid: ee.Algorithms.If(
                with_metrics.filter(ee.Filter.eq(model.class_property, cid)).size().gt(0),
                with_metrics.filter(ee.Filter.eq(model.class_property, cid)).aggregate_mean(
                    "_correct"
                ),
                None,
            )
        ),
    )
    per_class_n = ee.Dictionary.fromLists(
        ee_schema_names,
        ee_schema_ids.map(
            lambda cid: with_metrics.filter(ee.Filter.eq(model.class_property, cid)).size()
        ),
    )

    return {
        "overall_log_loss": with_metrics.aggregate_mean("_log_loss"),
        "overall_accuracy": with_metrics.aggregate_mean("_correct"),
        "per_class_log_loss": per_class_loss,
        "per_class_accuracy": per_class_acc,
        "n_samples": with_metrics.size(),
        "per_class_n_samples": per_class_n,
    }


def build_gee_probability_manifest(
    *,
    class_schema: ClassSchema,
    encoding: ProbabilityEncoding = "uint16",
    crs: str = "",
    epoch: str | None = None,
    source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ProbabilityManifest:
    """Build a validated :class:`ProbabilityManifest` matching a GEE-exported probability tile."""
    if encoding not in ("uint16", "float32"):
        raise ValueError(f"encoding must be 'uint16' or 'float32', got {encoding!r}.")
    return ProbabilityManifest(
        class_schema=class_schema,
        dtype=encoding,
        scale_factor=UINT16_PROB_SCALE if encoding == "uint16" else 1.0,
        nodata=UINT16_PROB_NODATA if encoding == "uint16" else None,
        crs=crs,
        epoch=epoch,
        paradigm="pixel",
        source=source,
        extra=dict(extra) if extra else {},
    )


def export_probability_tiles(
    prob_image: Any,
    bounds: tuple[float, float, float, float],
    *,
    export_prefix: str,
    class_schema: ClassSchema | None = None,
    config: ProjectConfig | None = None,
    encoding: ProbabilityEncoding = "uint16",
    scale_on_export: bool = True,
    epoch: str | None = None,
    tile_size_deg: float = 0.5,
    scale_m: float = 10.0,
    crs: str = "EPSG:4326",
    destination: Literal["drive", "asset"] = "drive",
    drive_folder: str = "LC_Framework",
    existing_assets: Iterable[str] = (),
    skip_existing: bool = True,
    dry_run: bool = False,
    manifest_dir: str | Path | None = None,
    pyramiding_policy: str = "mean",
    max_pixels: float = 1e13,
    extra: dict[str, Any] | None = None,
) -> GEEProbabilityExportResult:
    """Schedule tiled GEE Class Probability Stack exports paired with sidecar manifests.

    Expects continuous ``[0, 1]`` probabilities from :func:`predict_gee_probabilities`
    when ``scale_on_export=True`` (default); set ``scale_on_export=False`` if the input
    image was already scaled via :func:`scale_probability_image`.
    """
    schema = resolve_class_schema(class_schema, config, caller="export_probability_tiles")
    if destination not in ("drive", "asset"):
        raise ValueError(f"destination must be 'drive' or 'asset', got {destination!r}.")
    if scale_m <= 0:
        raise ValueError(f"scale_m must be > 0, got {scale_m!r}.")

    tiles = generate_tile_grid(bounds, tile_size_deg)
    plans = plan_tiled_gee_exports(
        tiles,
        export_prefix=export_prefix,
        existing_assets=existing_assets,
        skip_existing=skip_existing,
    )

    manifests: dict[str, ProbabilityManifest] = {}
    for plan in plans:
        manifests[plan.tile.tile_id] = build_gee_probability_manifest(
            class_schema=schema,
            encoding=encoding,
            crs=crs,
            epoch=epoch,
            source=plan.asset_id if destination == "asset" else f"{plan.description}.tif",
            extra=extra,
        )

    manifest_paths: dict[str, Path] = {}
    if dry_run:
        return GEEProbabilityExportResult(
            plans=plans,
            tasks=[],
            manifests=manifests,
            manifest_paths=manifest_paths,
        )

    ee = require_ee("GEE tiled probability export")
    ordered_prob = prob_image.select(list(schema.gee_prob_band_names))
    prepared = (
        scale_probability_image(ordered_prob, encoding=encoding)
        if scale_on_export
        else ordered_prob
    )

    def _make_task(plan: ExportTilePlan) -> Any:
        region = ee.Geometry.Rectangle(list(plan.tile.bounds))
        tile_img = prepared.clip(region)
        if destination == "asset":
            return ee.batch.Export.image.toAsset(
                image=tile_img,
                description=plan.description,
                assetId=plan.asset_id,
                region=region,
                crs=crs,
                scale=float(scale_m),
                maxPixels=max_pixels,
                pyramidingPolicy={".default": pyramiding_policy},
            )

        if encoding == "uint16":
            # Explicitly unmask to uint16 nodata (65535) so masked pixels never blend into 0
            drive_img = tile_img.unmask(UINT16_PROB_NODATA).toUint16()
            fmt_opts: dict[str, Any] = {"cloudOptimized": True, "noData": UINT16_PROB_NODATA}
        else:
            drive_img = tile_img.toFloat()
            fmt_opts = {"cloudOptimized": True}

        return ee.batch.Export.image.toDrive(
            image=drive_img,
            description=plan.description,
            folder=drive_folder,
            fileNamePrefix=plan.description,
            region=region,
            crs=crs,
            scale=float(scale_m),
            maxPixels=max_pixels,
            formatOptions=fmt_opts,
        )

    tasks = submit_gee_export_tasks(plans, task_factory=_make_task, dry_run=False)

    if manifest_dir is not None:
        m_root = Path(manifest_dir)
        m_root.mkdir(parents=True, exist_ok=True)
        for plan in plans:
            if plan.skip:
                continue
            mpath = manifest_path_for(m_root / f"{plan.description}.tif")
            manifests[plan.tile.tile_id].to_json(mpath)
            manifest_paths[plan.tile.tile_id] = mpath

    return GEEProbabilityExportResult(
        plans=plans,
        tasks=tasks,
        manifests=manifests,
        manifest_paths=manifest_paths,
    )
