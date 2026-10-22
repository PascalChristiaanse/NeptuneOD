#!/usr/bin/env python
"""Serve a pickled Matplotlib figure with matplotlib's WebAgg backend.

This is a small helper for the ``fig-viewer`` VS Code extension. It loads a
figure pickled by ``orbitdet.visualization.base.Plot._save_figure_pickle`` (a
raw ``pickle.dump`` of a :class:`matplotlib.figure.Figure`), attaches a WebAgg
canvas/manager to it and serves the *interactive* figure over HTTP on the
loopback interface.

Once the server is listening it prints a single line to stdout and flushes it::

    FIG_VIEWER_READY http://127.0.0.1:<port>/<fignum>

and then blocks, serving the figure until the process is terminated.

The output is machine-readable on line one; anything the extension should show
to the user on failure is written to stderr and the process exits non-zero.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import sys


def find_free_port(host: str) -> int:
    """Ask the OS for an unused TCP port on *host*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", help="Path to a pickled Matplotlib figure (.fig).")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1, loopback only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="TCP port to use; 0 (default) selects a free port automatically.",
    )
    args = parser.parse_args(argv)

    # Import matplotlib only after argument parsing so ``--help`` stays fast and
    # free of backend side effects.
    import matplotlib

    matplotlib.use("WebAgg", force=True)
    # Do not try to launch an external browser; we only want the HTTP server.
    matplotlib.rcParams["webagg.open_in_browser"] = False

    from matplotlib import _pylab_helpers
    from matplotlib.backends import backend_webagg as webagg

    with open(args.figure, "rb") as handle:
        figure = pickle.load(handle)

    # Give the pickled figure a live WebAgg canvas/manager. Creating the
    # manager does NOT register it with the global figure manager (Gcf), which
    # the tornado request handlers use to find the figure by number. Without
    # registering it here, the page and toolbar load but the websocket finds no
    # manager and closes immediately, leaving an empty canvas.
    manager = webagg.new_figure_manager_given_figure(1, figure)
    _pylab_helpers.Gcf._set_new_active_manager(manager)

    port = args.port or find_free_port(args.host)
    # ``WebAggApplication.initialize`` reads these rcParams (its ``port``
    # argument is ignored), so set them here to bind the interface/port we want.
    # A small retry window lets matplotlib roll forward if the candidate port is
    # taken between our probe and the bind; the bound port is read back below.
    matplotlib.rcParams["webagg.address"] = args.host
    matplotlib.rcParams["webagg.port"] = port
    matplotlib.rcParams["webagg.port_retries"] = 10
    webagg.WebAggApplication.initialize()

    url = f"http://{webagg.WebAggApplication.address}:{webagg.WebAggApplication.port}/{manager.num}"
    print(f"FIG_VIEWER_READY {url}", flush=True)

    # Blocks until the process is terminated (SIGINT/SIGTERM).
    webagg.WebAggApplication.start()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - surfaced to the user by the extension
        print(f"fig-server error: {exc}", file=sys.stderr, flush=True)
        raise
