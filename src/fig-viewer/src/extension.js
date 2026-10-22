"use strict";

const vscode = require("vscode");
const { FigEditorProvider, VIEW_TYPE } = require("./figEditorProvider");

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  const log = vscode.window.createOutputChannel("FIG Viewer");
  context.subscriptions.push(log);

  const provider = new FigEditorProvider(context, log);

  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(VIEW_TYPE, provider, {
      // Keep the Python server alive while the tab is hidden so switching
      // between figures is instant.
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: false,
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("figViewer.openPreview", async (uri) => {
      const target = uri || vscode.window.activeTextEditor?.document.uri;
      if (!target) {
        vscode.window.showWarningMessage("FIG Viewer: no .fig file selected.");
        return;
      }
      await vscode.commands.executeCommand("vscode.openWith", target, VIEW_TYPE);
    })
  );
}

function deactivate() {
  // Child processes are stopped through the document lifecycle.
}

module.exports = { activate, deactivate };
