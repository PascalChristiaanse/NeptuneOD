# FIG Viewer — VS Code extension

Open pickled Matplotlib `.fig` files (the files written by
`orbitdet.visualization.base.Plot._save_figure_pickle`) directly in VS Code as a
**read-only, fully interactive** figure.

## How it works

A `.fig` file is a raw `pickle.dump(fig)`. VS Code's webviews run JavaScript,
so they cannot unpickle Python objects. Instead:

1. The custom editor spawns `python/fig_server.py` with your configured
   interpreter.
2. That helper unpickles the figure, attaches a live WebAgg canvas/manager to
   it, and serves matplotlib's own interactive page (pan, zoom, toolbar, and the
   `format_coord` hover readout) over HTTP on `127.0.0.1`.
3. The webview embeds that loopback page in an `<iframe>`.

From VS Code's perspective it is a read-only custom editor; interactivity is
handled by matplotlib inside the iframe.

```
fig-viewer/
├── package.json           # extension manifest (custom editor contribution)
├── python/fig_server.py   # unpickles + serves the figure via WebAgg
└── src/
    ├── extension.js       # activate/deactivate + command
    ├── figEditorProvider.js
    ├── figServer.js       # child-process lifecycle
    └── pythonResolver.js  # interpreter resolution
```

## Requirements

- `matplotlib` **with `tornado`** installed in the Python environment used to
  render (matplotlib's WebAgg backend requires tornado; conda's `matplotlib`
  includes it).

## Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `figViewer.pythonPath` | `""` | Explicit Python interpreter path. Takes priority. |
| `figViewer.condaEnv` | `"NeptuneOD"` | Conda env resolved via `conda run -n <env> python`. |
| `figViewer.extraPythonArgs` | `[]` | Extra args before the server script (e.g. `-u`). |

Precedence: `pythonPath` → `condaEnv` → workspace `.venv`/`venv` → `python3`.

## Build / run (development)

No build step is required — the extension is plain JavaScript.

1. Open the `src/fig-viewer` folder in VS Code.
2. Press **F5** (Run → Start Debugging) to launch an Extension Development Host.
3. In the new window, open any `.fig` file. It opens in the FIG Viewer.
   - Or run **FIG Viewer: Open with Interactive Viewer** from the command
     palette while a `.fig` is focused.

To produce an installable package, install `@vscode/vsce` and run:

```bash
npx @vscode/vsce package
code --install-extension fig-viewer-0.0.1.vsix
```

## Test the Python helper directly

```bash
python src/fig-viewer/python/fig_server.py path/to/figure.fig --port 8977
# prints:  FIG_VIEWER_READY http://127.0.0.1:8977/1
# then open that URL in a browser
```

## Troubleshooting

- **"Could not render figure"** — check the **FIG Viewer** output channel. The
  most common cause is a Python interpreter without `matplotlib`, or one where
  `tornado` is missing.
- Each open `.fig` gets its own short-lived Python process; it is stopped when
  the editor tab is closed.
