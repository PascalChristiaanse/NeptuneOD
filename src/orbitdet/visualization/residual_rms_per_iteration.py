import matplotlib.pyplot as plt
import numpy as np

from orbitdet.visualization.estimation_base import EstimationPlot


class ResidualRMSPerIteration(EstimationPlot):
    """Plot the RMS of the residuals per iteration of the estimation process.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
    estimation_output : est_an.EstimationOutput
        The estimation output whose ``residual_history`` is plotted.
    """

    def _make_figure(self):
        estimation_output = self.estimation_output

        rms_per_iteration = []
        for iteration in range(estimation_output.residual_history.shape[1]):
            residuals = estimation_output.residual_history[:, iteration]
            rms = np.sqrt(np.mean(residuals**2))
            rms_per_iteration.append(rms)

        fig, ax = plt.subplots()
        ax.plot(rms_per_iteration, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMS of Residuals")
        ax.set_title("RMS of Residuals per Iteration")
        ax.grid()
        return fig, ax
