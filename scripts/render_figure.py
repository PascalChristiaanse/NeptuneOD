import pickle
import sys
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")

def render_figure(input_path: Path) -> None:
    with input_path.open("rb") as f:
        fig = pickle.load(f)
    plt.show()

if __name__ == "__main__":
    render_figure(
        input_path=Path(sys.argv[1]),
    )