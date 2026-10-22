"use strict";

const fs = require("fs");
const path = require("path");
const vscode = require("vscode");

/**
 * Work out how to launch a Python interpreter that can unpickle `.fig` files.
 * Precedence:
 *   1. `figViewer.pythonPath`   - explicit interpreter path.
 *   2. `figViewer.condaEnv`     - `conda run -n <env> python`.
 *   3. workspace `.venv`/`venv` - local virtualenv.
 *   4. `python3` on the PATH.
 * @param {string|undefined} workspaceFolder
 * @returns {{command: string, args: string[]}}
 */
function resolvePythonCommand(workspaceFolder) {
  const config = vscode.workspace.getConfiguration("figViewer");

  const configuredPath = (config.get("pythonPath", "") || "").trim();
  if (configuredPath) {
    return { command: configuredPath, args: [] };
  }

  const condaEnv = (config.get("condaEnv", "") || "").trim();
  if (condaEnv) {
    const conda = findCondaExecutable();
    if (conda) {
      // `--no-capture-output` keeps stdout streaming so the READY line is
      // observed promptly instead of being buffered until the server exits.
      return {
        command: conda,
        args: ["run", "--no-capture-output", "-n", condaEnv, "python"],
      };
    }
  }

  if (workspaceFolder) {
    for (const relative of [".venv/bin/python", "venv/bin/python"]) {
      const candidate = path.join(workspaceFolder, relative);
      if (fs.existsSync(candidate)) {
        return { command: candidate, args: [] };
      }
    }
  }

  return { command: "python3", args: [] };
}

/** @returns {string|undefined} */
function findCondaExecutable() {
  const fromEnv = process.env.CONDA_EXE;
  if (fromEnv && fs.existsSync(fromEnv)) {
    return fromEnv;
  }
  const home = process.env.HOME || process.env.USERPROFILE || "";
  const candidates = [
    path.join(home, "miniconda3", "bin", "conda"),
    path.join(home, "anaconda3", "bin", "conda"),
    "/opt/conda/bin/conda",
  ];
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return "conda"; // fall back to PATH resolution
}

module.exports = { resolvePythonCommand };
