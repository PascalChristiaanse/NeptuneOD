"use strict";

const path = require("path");
const vscode = require("vscode");
const { FigServer } = require("./figServer");

const VIEW_TYPE = "figViewer.figEditor";

/**
 * Read-only custom editor for pickled Matplotlib figures (`.fig`).
 *
 * VS Code cannot unpickle a Python object in a webview, so the editor spawns a
 * helper Python process that loads the figure and serves it with matplotlib's
 * WebAgg backend on loopback. The webview embeds that page in an iframe, giving
 * full interactive pan/zoom/hover while remaining, from VS Code's point of
 * view, a read-only editor.
 */
class FigEditorProvider {
  /**
   * @param {vscode.ExtensionContext} context
   * @param {vscode.OutputChannel} log
   */
  constructor(context, log) {
    this.context = context;
    this.log = log;
    /** @type {Map<string, FigServer>} */
    this.servers = new Map();
  }

  openCustomDocument(uri /*, openContext, token */) {
    return { uri, dispose() {} };
  }

  async resolveCustomEditor(document, webviewPanel /*, token */) {
    webviewPanel.webview.options = { enableScripts: true };
    const key = document.uri.toString();

    let server = this.servers.get(key);
    if (!server) {
      const scriptPath = path.join(
        this.context.extensionPath,
        "python",
        "fig_server.py"
      );
      server = new FigServer(document.uri.fsPath, scriptPath, this.log);
      this.servers.set(key, server);
    }

    webviewPanel.webview.html = loadingHtml();

    if (server.url) {
      webviewPanel.webview.html = viewerHtml(server.url);
    }

    try {
      const url = server.url || (await server.ready);
      webviewPanel.webview.html = viewerHtml(url);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.log.appendLine(`[fig-viewer] error: ${message}`);
      webviewPanel.webview.html = errorHtml(message);
    }

    webviewPanel.onDidDispose(() => {
      this.servers.delete(key);
      if (server) {
        server.stop();
      }
    });
  }
}

function loadingHtml() {
  return wrap(
    `<div class="msg">
      <div class="spinner"></div>
      <p>Starting figure server…</p>
    </div>`
  );
}

function errorHtml(message) {
  return wrap(
    `<div class="msg error">
      <h2>Could not render figure</h2>
      <pre>${escapeHtml(message)}</pre>
      <p>Check the <b>FIG Viewer</b> output channel, and verify that
      <code>figViewer.pythonPath</code> (or <code>figViewer.condaEnv</code>)
      points at an interpreter with <code>matplotlib</code> installed.</p>
    </div>`
  );
}

function viewerHtml(url) {
  const style =
    "position:absolute;inset:0;width:100%;height:100%;border:0;background:#fff;";
  return wrap(
    `<iframe src="${escapeHtml(url)}" style="${style}"
        sandbox="allow-scripts allow-same-origin allow-downloads"></iframe>`
  );
}

function wrap(body) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta http-equiv="Content-Security-Policy"
      content="default-src none; frame-src http://127.0.0.1:* http://localhost:*; style-src 'unsafe-inline';" />
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden;
    background: var(--vscode-editor-background);
    color: var(--vscode-editor-foreground);
    font-family: var(--vscode-font-family); }
  .msg { display: flex; flex-direction: column; align-items: center;
    justify-content: center; height: 100%; gap: 12px; text-align: center;
    padding: 24px; box-sizing: border-box; }
  .msg pre { text-align: left; white-space: pre-wrap; max-width: 720px;
    background: var(--vscode-textCodeBlock-background); padding: 12px;
    border-radius: 6px; overflow: auto; }
  .spinner { width: 32px; height: 32px; border-radius: 50%;
    border: 3px solid var(--vscode-foreground); border-top-color: transparent;
    animation: spin 0.9s linear infinite; opacity: 0.6; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>${body}</body>
</html>`;
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

module.exports = { FigEditorProvider, VIEW_TYPE };
