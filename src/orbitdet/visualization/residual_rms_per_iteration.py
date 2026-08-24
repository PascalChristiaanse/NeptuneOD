import matplotlib.pyplot as plt
import numpy as np

from orbitdet.visualization.base import Plot


class ResidualRMSPerIteration(Plot):
    """Plot the RMS of the residuals per iteration of the estimation process."""

    def __init__(
        self,
        cfg,
        estimation_output
    ):
        super().__init__(cfg)
        self.estimation_output = estimation_output

    def _make_figure(self):
        estimation_output = self.estimation_output

        """
        Plots the RMS of the residuals per iteration of the estimation process.

        Parameters
        ----------
        estimation_output : est_an.EstimationOutput
            The output of the estimation process containing the residuals per iteration.
        """

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
