# FIG Viewer

A VS Code extension that opens pickled Matplotlib figures (`.fig` files) in a
**read-only, fully interactive** editor — pan, zoom, the standard matplotlib
toolbar, and the live coordinate readout, all inside VS Code.

No re-running of your plotting code, no external browser window: double-click a
`.fig` and it opens in the editor beside your source.

<!-- Replace OWNER/REPO once this lives in its own repository. -->
<!-- [![Visual Studio Marketplace](https://img.shields.io/badge/marketplace-FIG%20Viewer-blue)](https://marketplace.visualstudio.com/items?itemName=neptuneod.fig-viewer) -->

---

## Why this exists

Matplotlib's `.fig` format is just a `pickle.dump(fig)` of a
`matplotlib.figure.Figure`. It is the natural way to preserve an *interactive*
figure for later — a `.png` is flat, and re-running the script is expensive or
non-deterministic. But nothing in VS Code knows how to open a Python pickle.

This extension bridges that gap. VS Code webviews run JavaScript and cannot
unpickle a Python object, so the extension delegates the hard part to Python and
embeds the result:

1. A **custom read-only editor** is registered for `*.fig`.
2. Opening a `.fig` spawns a short-lived helper process
   (`python/fig_server.py`) with your configured interpreter.
3. The helper unpickles the figure, attaches a live
   [WebAgg](https://matplotlib.org/stable/users/explain/figure/webagg.html)
   canvas/manager to it, and serves matplotlib's own interactive page over HTTP
   on `127.0.0.1`.
4. The webview embeds that loopback page in an `<iframe>`.

From VS Code's point of view the editor is read-only; all interactivity is
matplotlib's, running in the iframe.

```
┌──────────────┐   spawn   ┌──────────────────┐   HTTP/WS   ┌───────────┐
│ VS Code      │ ────────▶ │ fig_server.py    │ ──────────▶ │ <iframe>  │
│ custom editor│           │ (unpickle+WebAgg)│  loopback   │ in webview│
└──────────────┘           └──────────────────┘             └───────────┘
```

---

## Features

- Opens `*.fig` automatically — or via **FIG Viewer: Open with Interactive
  Viewer** from the command palette.
- Full matplotlib WebAgg interactivity: pan, zoom, home/back/forward, save,
  and the hover coordinate readout (`format_coord`).
- Read-only: the `.fig` on disk is never modified.
- One isolated Python process per open figure, cleaned up when the tab closes.
- Configurable interpreter (explicit path, conda env, workspace virtualenv, or
  `python3`).

## Requirements

- **VS Code** 1.90 or newer.
- **Python 3** with **`matplotlib`** installed, including its **`tornado`**
  dependency (required by the WebAgg backend).

If you install matplotlib with conda, `tornado` is included. If you use pip,
`pip install matplotlib` pulls in tornado as well.

> **Note on `.fig` producers:** any code that does `pickle.dump(fig, f)` works.
> The figures must be picklable, which means the figure must not hold
> un-picklable callables (e.g. nested-function closures or lambdas assigned to
> `ax.format_coord`, tick formatters, etc.). See
> [Picklable figures](#picklable-figures) below.

---

## Installation

### Option 1 — Install from a `.vsix` (recommended)

A `.vsix` is the standard installable bundle and works on any machine, local or
remote.

1. Obtain or build the package (see [Building a VSIX](#building-a-vsix)).
2. Install it:

   ```bash
   # Local VS Code
   code --install-extension fig-viewer-0.0.1.vsix

   # Remote-SSH / WSL / dev container: install into the remote extension host
   code --install-extension fig-viewer-0.0.1.vsix --force
   ```

   Or from the UI: open the **Extensions** view, click the `…` menu, choose
   **Install from VSIX…**, select the file, then reload the window.

### Option 2 — Copy the folder (no tooling required)

VS Code loads any extension from a folder named
`<publisher>.<name>-<version>` inside its extensions directory, so you can drop
the source in by hand:

1. Copy this folder to the extensions directory for your setup:

   | Scenario | Extensions directory |
   | --- | --- |
   | Local VS Code (Linux/macOS) | `~/.vscode/extensions/` |
   | Local VS Code (Windows) | `%USERPROFILE%\.vscode\extensions\` |
   | Remote-SSH / WSL / dev container | `~/.vscode-server/extensions/` |
   | VS Code Insiders | `~/.vscode-insiders/extensions/` |

2. Name the destination `neptuneod.fig-viewer-0.0.1` — the folder name must be
   `<publisher>.<name>-<version>` and match `package.json`.

   ```bash
   cp -r fig-viewer ~/.vscode/extensions/neptuneod.fig-viewer-0.0.1
   ```

3. Run **Developer: Reload Window**.

VS Code's extension manager may show this as an "unpacked" extension; that is
expected for personal use.

### Option 3 — Run from source (for development)

1. Open this folder in VS Code.
2. Press **F5** (**Run → Start Debugging**) — this launches an
   **Extension Development Host** with the extension loaded.
3. In that new window, open any `.fig`.

There is **no build step**: the extension is plain JavaScript and loads directly.

---

## Usage

- **Just open it:** double-click a `.fig` file, or use **File → Open**. It opens
  in the FIG Viewer automatically.
- **Command palette:** with a `.fig` focused, run
  **FIG Viewer: Open with Interactive Viewer**.
- **Choose editor explicitly:** right-click a `.fig` → **Open With…** →
  **Matplotlib Figure**.
- An editor tab labelled with the figure name appears with matplotlib's toolbar
  and canvas.

## Configuration

Settings live under `figViewer.*` (workspace or user `settings.json`):

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `figViewer.pythonPath` | `string` | `""` | Absolute path to the Python interpreter used to load `.fig` files. Highest priority. |
| `figViewer.condaEnv` | `string` | `"NeptuneOD"` | Conda environment name, resolved as `conda run -n <env> python`. Used when `pythonPath` is empty. |
| `figViewer.extraPythonArgs` | `string[]` | `[]` | Extra arguments inserted before the server script (e.g. `["-u"]` for unbuffered output). |

**Interpreter resolution order:**

1. `figViewer.pythonPath`
2. `figViewer.condaEnv` (via `conda run`)
3. A workspace virtualenv at `.venv/` or `venv/`
4. `python3` on the `PATH`

Example `settings.json`:

```jsonc
{
  // Point directly at an interpreter…
  "figViewer.pythonPath": "/home/you/miniconda3/envs/NeptuneOD/bin/python"

  // …or name the conda environment instead (omit pythonPath):
  // "figViewer.condaEnv": "NeptuneOD"
}
```

The target interpreter only needs `matplotlib` (with `tornado`); it does **not**
need to be the same interpreter that created your project.

---

## Command-line helper

The same server can be run standalone, outside VS Code:

```bash
python python/fig_server.py path/to/figure.fig --port 8977
```

It prints a single ready line and then serves the figure:

```
FIG_VIEWER_READY http://127.0.0.1:8977/1
Press Ctrl+C to stop WebAgg server
```

Open that URL in any browser. Options: `--host` (default `127.0.0.1`, loopback
only) and `--port` (`0` = pick a free port).

---

## Troubleshooting

- **Editor shows "Could not render figure".** Open the **FIG Viewer** output
  channel (`View → Output`, then select *FIG Viewer*). The exact Python error is
  logged there. The most common causes:
  - the interpreter has no `matplotlib`, or
  - it has `matplotlib` but not `tornado`.
- **Blank canvas / toolbar but no figure.** This usually means the helper
  process could not attach the figure to the WebAgg server. Check the output
  channel for a traceback from `fig_server.py`.
- **Wrong interpreter.** Verify with `figViewer.pythonPath`, or check the
  command logged at the top of the output channel (it prints the exact
  executable and arguments used).
- **Figure fails to unpickle.** The figure contains an un-picklable object.
  See [Picklable figures](#picklable-figures).
- **Processes left behind.** Each figure's process exits when its editor tab
  closes; if VS Code is killed forcibly, a stray `fig_server.py` may linger —
  `pkill -f fig_server.py` clears it.

### Picklable figures

`pickle` cannot serialize nested-function closures or lambdas. If a figure
stores one — most commonly via `ax.format_coord = some_closure` or a
`FuncFormatter(lambda …)` — it will fail to load. Keep such callables
**module-level** and bind their arguments with `functools.partial`:

```python
import functools

# Bad: a nested closure cannot be pickled.
def make_formatter(x_label, y_label):
    def _format(x, y):
        return f"{x_label}: {x}, {y_label}: {y}"
    return _format

# Good: a module-level function wrapped in a picklable partial.
def _format(x, y, x_label, y_label):
    return f"{x_label}: {x}, {y_label}: {y}"

formatter = functools.partial(_format, x_label="x", y_label="y")
ax.format_coord = formatter
```

---

## Development

**Layout**

```
fig-viewer/
├── package.json              # extension manifest (customEditors, settings, command)
├── src/
│   ├── extension.js          # activate/deactivate + command registration
│   ├── figEditorProvider.js  # CustomReadonlyEditorProvider -> webview iframe
│   ├── figServer.js          # child-process lifecycle + readiness handshake
│   └── pythonResolver.js     # interpreter resolution
├── python/
│   └── fig_server.py         # unpickles the figure and serves it via WebAgg
└── README.md
```

**The readiness handshake.** `fig_server.py` prints `FIG_VIEWER_READY <url>` on
stdout once it is listening; `figServer.js` waits for that exact prefix before
loading the iframe, and treats any early exit as an error surfaced to the user.

**No build step.** The extension is CommonJS JavaScript executed by the VS Code
extension host's own Node runtime. Nothing to compile.

**Debugging.** Press **F5** to launch an Extension Development Host (a
`.vscode/launch.json` is included). Logs from both the extension and the Python
helper stream into the **FIG Viewer** output channel.

### Building a VSIX

Requires Node.js/npm on the build machine:

```bash
npx @vscode/vsce package        # -> fig-viewer-0.0.1.vsix
```

### Publishing

To publish to the Marketplace or [Open VSX](https://open-vsx.org/), add a
`repository`, an `icon`, and a publisher/PAT, then:

```bash
npx @vscode/vsce publish        # Marketplace
npx ovsx publish                 # Open VSX
```

---

## Limitations

- Read-only. Editing a figure is out of scope; regenerate the `.fig` instead.
- Requires a Python environment with matplotlib available to VS Code.
- Loads the whole figure into memory of a helper process — identical to opening
  it with `python`.
- The helper binds to loopback (`127.0.0.1`) only; it is not exposed to the
  network.

## Contributing

Issues and pull requests are welcome. Please keep changes to
`python/fig_server.py` backend-agnostic where possible and add a note to the
README for any new setting.

## License

MIT.
