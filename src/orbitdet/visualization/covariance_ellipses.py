import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from tudatpy.astro import frame_conversion as fc
from tudatpy.dynamics import environment as env
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.reproducibility import RuntimeContext
from orbitdet.visualization.base import Plot


def _get_inertial_state_at_epoch(
    bodies: env.SystemOfBodies, body_name: str, central_body_name: str, epoch
) -> np.ndarray:
    """
    Retrieves the relative state (body w.r.t. central body) in the inertial
    frame at the given epoch.

    Uses the Ephemeris objects from the SystemOfBodies to get Cartesian states at the given epoch.

    Parameters
    ----------
    bodies : SystemOfBodies
        The system of bodies.
    body_name : str
        The name of the body whose state is needed.
    central_body_name : str
        The name of the central body.
    epoch : Time
        The epoch at which to evaluate the state.

    Returns
    -------
    np.ndarray
        6-element array [x, y, z, vx, vy, vz] in the inertial frame.
    """
    epoch_seconds = float(epoch.to_float())
    body_state = np.array(bodies.get(body_name).ephemeris.cartesian_state(epoch_seconds))
    central_state = np.array(bodies.get(central_body_name).ephemeris.cartesian_state(epoch_seconds))
    return body_state - central_state


def _rotate_covariance_to_rsw(covariance: np.ndarray, inertial_state: np.ndarray) -> np.ndarray:
    """
    Rotates a covariance matrix from inertial to RSW frame.

    Only the first 6 parameters (Cartesian position + velocity) are rotated into the
    RSW frame. All other parameters (e.g. gravitational parameter, gravity field
    coefficients) are left unchanged. Cross-correlation terms between the state and
    other parameters are properly rotated.

    Parameters
    ----------
    covariance : np.ndarray
        NxN covariance matrix in inertial frame. The first 6 parameters must
        correspond to [x, y, z, vx, vy, vz].
    inertial_state : np.ndarray
        6-element inertial state vector [x, y, z, vx, vy, vz] defining the RSW frame.

    Returns
    -------
    np.ndarray
        NxN covariance matrix with the first 6 parameters rotated into the RSW frame.
    """
    n_params = covariance.shape[0]
    r_inertial_to_rsw = fc.inertial_to_rsw_rotation_matrix(inertial_state)

    # Build block-diagonal 6x6 rotation matrix for state parameters
    rotation_6 = np.zeros((6, 6))
    rotation_6[:3, :3] = r_inertial_to_rsw
    rotation_6[3:, 3:] = r_inertial_to_rsw

    if n_params == 6:
        return rotation_6 @ covariance @ rotation_6.T

    # For N > 6: build a full NxN block-diagonal rotation matrix
    # with identity for non-state parameters
    rotation_n = np.eye(n_params)
    rotation_n[:6, :6] = rotation_6

    return rotation_n @ covariance @ rotation_n.T


class CovarianceEllipses(Plot):
    """Plot a grid of pairwise uncertainty ellipses from the estimation covariance matrix."""

    def __init__(
        self,
        cfg,
        estimation_output: est_an.EstimationOutput,
        bodies: env.SystemOfBodies,
        ctx: RuntimeContext,
    ):
        super().__init__(cfg)
        self.estimation_output = estimation_output
        self.bodies = bodies
        self.ctx = ctx

    def _make_figure(self):
        cfg = self.cfg
        estimation_output = self.estimation_output
        bodies = self.bodies
        ctx = self.ctx

        """
        Plots a grid of pairwise uncertainty ellipses from the estimation covariance matrix.

        Parameters
        ----------
        cfg : DictConfig
            Configuration containing optional 'covariance_ellipses' settings.
            Supported keys:
                rotate_to_rsw : bool
                    Whether to rotate the covariance to RSW frame (default True).
                    Only the first 6 parameters (Cartesian state) are rotated; all
                    other parameters (GM, gravity field coefficients, etc.) remain unchanged.
                body_name : str
                    Name of the body whose state defines the RSW frame (default "Triton").
                central_body_name : str
                    Name of the central body (default "Neptune").
                parameter_names : list of str
                    Names for each parameter. Defaults to RSW frame names for the first
                    6 parameters; additional parameters use auto-generated names.
                n_sigma : float
                    Number of sigmas for ellipse scaling (default 1.0).
                figure : dict
                    'width' and 'height' per subplot.
                confidence : bool
                    Whether to annotate ellipses with confidence percentage (default False).

        estimation_output : est_an.EstimationOutput
            The output of the estimation process containing the covariance matrix.
        bodies : SystemOfBodies
            The system of bodies (used to get the state for RSW rotation).
        ctx : RuntimeContext
            Runtime context providing the epoch for state evaluation.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : np.ndarray
            Array of axes objects.
        """
        plot_cfg = cfg.get("covariance_ellipses", {})

        covariance = estimation_output.covariance
        n_params = covariance.shape[0]

        # Default parameter names: RSW frame
        default_names = [
            "R",
            "S",
            "W",
            "v_R",
            "v_S",
            "v_W",
        ]
        parameter_names = plot_cfg.get("parameter_names", default_names)

        # Extend or truncate parameter_names to match matrix size
        if len(parameter_names) < n_params:
            parameter_names = list(parameter_names) + [
                f"p{i}" for i in range(len(parameter_names), n_params)
            ]
        else:
            parameter_names = parameter_names[:n_params]

        # Optionally rotate to RSW frame
        rotate_to_rsw = plot_cfg.get("rotate_to_rsw", True)
        if rotate_to_rsw:
            body_name = plot_cfg.get("body_name", "Triton")
            central_body_name = plot_cfg.get("central_body_name", "Neptune")
            inertial_state = _get_inertial_state_at_epoch(
                bodies, body_name, central_body_name, ctx.start_epoch
            )
            plot_covariance = _rotate_covariance_to_rsw(covariance, inertial_state)
        else:
            plot_covariance = covariance

        # Extract formal errors (1-sigma) from diagonal

        n_sigma = plot_cfg.get("n_sigma", 1.0)
        confidence = plot_cfg.get("confidence", False)

        # Number of pairwise subplots: n_params choose 2
        n_pairs = n_params * (n_params - 1) // 2
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols if n_pairs > 0 else 1

        subplot_width = plot_cfg.get("figure", {}).get("width", 4.0)
        subplot_height = plot_cfg.get("figure", {}).get("height", 4.0)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * subplot_width, n_rows * subplot_height),
            squeeze=False,
        )

        pair_idx = 0
        for i in range(n_params):
            for j in range(i + 1, n_params):
                row = pair_idx // n_cols
                col = pair_idx % n_cols
                ax = axes[row, col]

                # Extract 2x2 sub-covariance matrix for parameters (i, j)
                sub_cov = np.array(
                    [
                        [plot_covariance[i, i], plot_covariance[i, j]],
                        [plot_covariance[j, i], plot_covariance[j, j]],
                    ]
                )

                # Compute eigenvalues and eigenvectors for ellipse
                eigenvalues, eigenvectors = np.linalg.eigh(sub_cov)

                # Semi-axes lengths (n_sigma * sqrt(eigenvalue))
                a = n_sigma * np.sqrt(eigenvalues[1])  # major axis
                b = n_sigma * np.sqrt(eigenvalues[0])  # minor axis

                # Rotation angle of the ellipse
                angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

                # Draw ellipse centered at origin
                ellipse = Ellipse(
                    xy=(0, 0),
                    width=2 * a,
                    height=2 * b,
                    angle=angle,
                    facecolor="steelblue",
                    edgecolor="navy",
                    alpha=0.4,
                    linewidth=1.5,
                )
                ax.add_patch(ellipse)

                # Set axis limits with some padding
                max_extent = max(a, b) * 1.3
                ax.set_xlim(-max_extent, max_extent)
                ax.set_ylim(-max_extent, max_extent)
                ax.set_aspect("equal")

                ax.set_xlabel(parameter_names[i])
                ax.set_ylabel(parameter_names[j])
                ax.set_title(f"{parameter_names[i]} vs {parameter_names[j]}")
                ax.grid(True, alpha=0.3)

                # Annotate with confidence if requested
                if confidence:
                    from scipy.stats import chi2

                    # 2D confidence: 1-sigma ≈ 39.3%, 2-sigma ≈ 86.5%, 3-sigma ≈ 98.9%
                    conf_1sigma = chi2.cdf(1.0**2, df=2) * 100
                    ax.text(
                        0.02,
                        0.98,
                        f"{conf_1sigma:.1f}%",
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.8),
                    )

                pair_idx += 1

        # Hide unused subplots
        for idx in range(pair_idx, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        # Compute eigendecomposition of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(plot_covariance)
        # Sort in descending order (largest uncertainty first)
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        # Condition index: sqrt(lambda_max / lambda_i)
        # Clip eigenvalues to avoid sqrt of negative values from numerical precision
        eigvals_safe = np.maximum(eigvals, 0.0)
        # Avoid divide-by-zero when eigenvalues are zero
        with np.errstate(divide="ignore", invalid="ignore"):
            condition_index = np.sqrt(eigvals_safe[0] / eigvals_safe)
        condition_index = np.where(np.isfinite(condition_index), condition_index, np.inf)

        # Build tables at the bottom of the figure
        fig.suptitle(
            f"Pairwise Uncertainty Ellipses ({n_sigma}σ)"
            + (" [RSW frame]" if rotate_to_rsw else " [Inertial frame]"),
            fontsize=14,
            y=0.98,
        )

        # Table 1: Eigenvalue diagnostics (top row of bottom area)
        diag_ax = fig.add_axes([0.05, 0.11, 0.9, 0.08])
        diag_ax.axis("off")

        diag_col_labels = ["Mode", "Eigenvalue", "Std Dev", "Condition Index"]
        diag_table_data = []
        for i in range(n_params):
            diag_table_data.append(
                [
                    f"{i + 1}",
                    f"{eigvals[i]:.6e}",
                    f"{np.sqrt(eigvals_safe[i]):.6e}",
                    f"{condition_index[i]:.2f}" if np.isfinite(condition_index[i]) else "inf",
                ]
            )

        diag_table = diag_ax.table(
            cellText=diag_table_data,
            colLabels=diag_col_labels,
            cellLoc="center",
            loc="center",
        )
        diag_table.auto_set_font_size(False)
        diag_table.set_fontsize(9)
        diag_table.scale(1.0, 1.5)

        # Style header row
        for col_idx in range(len(diag_col_labels)):
            diag_table[0, col_idx].set_facecolor("#4472C4")
            diag_table[0, col_idx].set_text_props(color="white", fontweight="bold")

        # Highlight the weakest mode (largest eigenvalue = first data row)
        for col_idx in range(len(diag_col_labels)):
            cell = diag_table[1, col_idx]
            cell.set_facecolor("#FFF2CC")
            cell.get_text().set_fontweight("bold")

        # Table 2: Eigenvector components (bottom row of bottom area)
        eig_ax = fig.add_axes([0.05, 0.02, 0.9, 0.08])
        eig_ax.axis("off")

        eig_col_labels = ["Mode"] + parameter_names
        eig_table_data = []
        for i in range(n_params):
            vec = eigvecs[:, i]
            row = [f"{i + 1}"]
            for k in range(n_params):
                sign = "+" if vec[k] >= 0 else "-"
                row.append(f"{sign}{abs(vec[k]):.3f}")
            eig_table_data.append(row)

        eig_table = eig_ax.table(
            cellText=eig_table_data,
            colLabels=eig_col_labels,
            cellLoc="center",
            loc="center",
        )
        eig_table.auto_set_font_size(False)
        eig_table.set_fontsize(9)
        eig_table.scale(1.0, 1.5)

        # Style header row
        for col_idx in range(len(eig_col_labels)):
            eig_table[0, col_idx].set_facecolor("#4472C4")
            eig_table[0, col_idx].set_text_props(color="white", fontweight="bold")

        # Highlight the weakest mode row
        for col_idx in range(len(eig_col_labels)):
            cell = eig_table[1, col_idx]
            cell.set_facecolor("#FFF2CC")
            cell.get_text().set_fontweight("bold")

        # Use constrained_layout instead of tight_layout to avoid conflicts
        # with manually added table axes
        fig.set_layout_engine("constrained")
        return fig, axes
