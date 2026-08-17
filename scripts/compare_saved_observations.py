"""Generate comparison plots for saved binary vs fresh observation sets.

For each binary file in SavedBinaryObservations/, loads it as an ObservationCollection,
finds the matching SingleObservationSet from the fresh YAML-built collection,
computes residuals for both, and plots them together on the same figure.
"""

import glob
import logging
import re
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig, OmegaConf
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup

from orbitdet.data import KernelManager
from orbitdet.observations import create_observation_collection
from orbitdet.observations.factory import create_observation_dataset
from orbitdet.reproducibility import RuntimeContext, initialize
from orbitdet.simulation import get_environment
from orbitdet.visualization.residuals import (
    _configure_datetime_axis,
    _rad_to_arcsec,
    _rms_arcsec,
    _seconds_since_j2000_to_datetimes,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

BINARY_DIR = Path("SavedBinaryObservations")
OUTPUT_DIR = Path("/tmp/obs_comparison")
FILENAME_RE = re.compile(r"Triton_(\d+)_(nm\d+)\.tudat$")


def _parse_filename(path: Path) -> tuple[str, str]:
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Cannot parse: {path.name}")
    return m.group(1), m.group(2)


@hydra.main(
    version_base=None, config_path="../conf", config_name="experiments/classic_triton_state"
)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ctx: RuntimeContext = initialize(cfg)
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)

    km = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()

    bodies = get_environment(cfg, ctx)

    datasets_cfg = OmegaConf.select(cfg, "datasets")

    # Pre-build simulators for all datasets (they share the same bodies)
    # We build each dataset one at a time to maintain explicit dataset_id → fresh_set mapping
    fresh_by_id: dict[str, obs.SingleObservationSet] = {}
    for ds_key, dsc in datasets_cfg.items():
        dataset, _ = create_observation_dataset(cfg, dsc, bodies)
        if isinstance(dataset, obs.SingleObservationSet):
            fresh_by_id[ds_key] = dataset
        elif isinstance(dataset, obs.ObservationCollection):
            sets = list(dataset.get_single_observation_sets())
            # Some datasets produce multiple sets (e.g. nm0019 for two observatories)
            # Store them with suffixed keys
            for i, s in enumerate(sets):
                k = f"{ds_key}_{i}" if i > 0 else ds_key
                fresh_by_id[k] = s
    logger.info(f"Built {len(fresh_by_id)} fresh sets from {len(datasets_cfg)} dataset configs")

    # Build combined collection + simulators for residual computation
    full_collection, observation_models = create_observation_collection(cfg, bodies)
    simulators = obs_sim_setup.create_observation_simulators(observation_models, bodies)

    # Compute residuals for each fresh set
    for fs in fresh_by_id.values():
        tmp = obs.ObservationCollection([fs])
        obs.compute_residuals_and_dependent_variables(tmp, simulators, bodies)
    logger.info("Computed pre-fit residuals")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build a map of binary files by dataset ID
    bin_files = sorted(glob.glob(str(BINARY_DIR / "Triton_*.tudat")))
    logger.info(f"Found {len(bin_files)} binary files")
    # Collect data for table
    table_data = []
    datasets_cfg = OmegaConf.select(cfg, "datasets")

    # Iterate over binary files, using their dataset ID to find the right fresh set
    for bf in bin_files:
        path = Path(bf)
        obs_code, ds_id = _parse_filename(path)

        # Look up the dataset config for this binary
        dsc = datasets_cfg.get(ds_id) if datasets_cfg else None
        if dsc is None:
            logger.warning(f"  No dataset config for {path.name}, skipping")
            table_data.append((obs_code, ds_id, "?", "—", "—", "no config"))
            continue

        ds_type = dsc.get("type", "?")

        bin_oc = obs.ObservationCollection.load_from_binary(str(path.with_suffix("")))
        bin_sos = list(bin_oc.get_single_observation_sets())[0]
        logger.info(f"  Loaded {path.name}: n={bin_sos.number_of_observables}")

        # Look up the fresh set by dataset ID (built from the same config)
        fresh_sos = fresh_by_id.get(ds_id)

        if fresh_sos is None:
            logger.warning(f"  No fresh match for {path.name}, skipping")
            table_data.append(
                (obs_code, ds_id, ds_type, bin_sos.number_of_observables, "—", "no match")
            )
            continue

        if bin_sos.number_of_observables == 0 or fresh_sos.number_of_observables == 0:
            logger.warning(
                f"  {path.name}: empty set (binary={bin_sos.number_of_observables}, "
                f"fresh={fresh_sos.number_of_observables}), skipping"
            )
            table_data.append(
                (
                    obs_code,
                    ds_id,
                    ds_type,
                    bin_sos.number_of_observables,
                    fresh_sos.number_of_observables,
                    "empty",
                )
            )
            continue

        fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        colors = ["#E24A33", "#348ABD"]
        markers = ["o", "s"]
        labels = ["Saved binary", "Fresh collection"]

        for idx, sos in enumerate([bin_sos, fresh_sos]):
            obs_times = _seconds_since_j2000_to_datetimes(
                np.array([float(t) for t in sos.observation_times])
            )
            res = np.array(sos.residuals)
            if res.ndim < 2 or res.shape[1] < 2:
                logger.warning(f"  {labels[idx]} for {path.name} has no residual data, skipping")
                continue
            ra_res = _rad_to_arcsec(res[:, 0])
            dec_res = _rad_to_arcsec(res[:, 1])
            ra_rms = _rms_arcsec(ra_res)
            dec_rms = _rms_arcsec(dec_res)

            axs[0].scatter(
                obs_times,
                ra_res,
                marker=markers[idx],
                s=30,
                color=colors[idx],
                alpha=0.6,
                label=f'{labels[idx]} — RMS={ra_rms:.3e}"',
            )
            axs[1].scatter(
                obs_times,
                dec_res,
                marker=markers[idx],
                s=30,
                color=colors[idx],
                alpha=0.6,
                label=f'{labels[idx]} — RMS={dec_rms:.3e}"',
            )

        target_name = bin_sos.link_definition.link_ends[
            next(iter(bin_sos.link_definition.link_ends.keys()))
        ].body_name

        axs[0].set_title(f"RA — {target_name}  ({obs_code}/{ds_id})")
        axs[1].set_title(f"Dec — {target_name}  ({obs_code}/{ds_id})")
        axs[0].set_ylabel("Residual [arcsec]")
        axs[1].set_ylabel("Residual [arcsec]")
        axs[1].set_xlabel("Epoch")
        _configure_datetime_axis(axs[0])
        _configure_datetime_axis(axs[1])
        axs[0].legend(fontsize=8)
        axs[1].legend(fontsize=8)

        fig.suptitle(f"Comparison: {ds_id} (obs code {obs_code}) — {target_name}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(OUTPUT_DIR / f"{obs_code}_{ds_id}.pdf")
        plt.close(fig)
        logger.info(f"  Saved {OUTPUT_DIR}/{obs_code}_{ds_id}.pdf")
        table_data.append(
            (
                obs_code,
                ds_id,
                ds_type,
                bin_sos.number_of_observables,
                fresh_sos.number_of_observables,
                "plotted",
            )
        )

    # Also check for binary files whose dataset ID is not in the YAML config
    for bf in bin_files:
        _, ds_id = _parse_filename(Path(bf))
        if ds_id not in datasets_cfg:
            logger.info(
                f"  Binary {bf} dataset {ds_id} not in YAML config,"
                " skipping (will be listed separately)"
            )

    # Print summary table
    print("\n" + "=" * 110)
    print(f"  OBSERVATION COUNT SUMMARY ({OUTPUT_DIR}/)")
    print("=" * 110)
    print(
        f"  {'Obs':>3}  {'Dataset':<10}  {'Type':<45}  {'Binary':>7}  {'Fresh':>7}  {'Status':<10}"
    )
    print(f"  {'─' * 3}  {'─' * 10}  {'─' * 45}  {'─' * 7}  {'─' * 7}  {'─' * 10}")
    for row in table_data:
        obs_code, ds_id, ds_type, n_bin, n_fresh, status = row
        b_str = str(n_bin) if isinstance(n_bin, int) else n_bin
        f_str = str(n_fresh) if isinstance(n_fresh, int) else n_fresh
        print(
            f"  {str(obs_code):>3}  {ds_id:<10}  {ds_type:<45}"
            f"  {b_str:>7}  {f_str:>7}  {status:<10}"
        )
    n_skipped = sum(1 for r in table_data if r[5] != "plotted")
    n_plotted = len(table_data) - n_skipped
    print(f"  {'─' * 3}  {'─' * 10}  {'─' * 45}  {'─' * 7}  {'─' * 7}  {'─' * 10}")
    print(f"  {n_plotted} plotted, {n_skipped} skipped  (total {len(table_data)} datasets in YAML)")
    print(f"{'=' * 110}")
    # List binaries not in YAML
    yaml_ds_ids = set(datasets_cfg.keys())
    extra_bins = [bf for bf in bin_files if _parse_filename(Path(bf))[1] not in yaml_ds_ids]
    if extra_bins:
        print("\n  Binary files without matching YAML config:")
        for bf in extra_bins:
            obs_c, ds_id = _parse_filename(Path(bf))
            print(f"    {Path(bf).name}  (obs code {obs_c})")

    logger.info(f"All plots in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
