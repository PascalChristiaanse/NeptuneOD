"use strict";

const { spawn } = require("child_process");
const readline = require("readline");
const vscode = require("vscode");
const { resolvePythonCommand } = require("./pythonResolver");

/** Startup line the Python server prints once it is serving. */
const READY_PREFIX = "FIG_VIEWER_READY ";

/**
 * Owns a child Python process running `python/fig_server.py` for one `.fig`
 * document and exposes an async `ready` promise resolving to the loopback URL
 * of the interactive figure.
 *
 * All cleanup is idempotent so no orphan Python processes are left behind.
 */
class FigServer {
  /**
   * @param {string} figurePath
   * @param {string} scriptPath
   * @param {vscode.OutputChannel} log
   */
  constructor(figurePath, scriptPath, log) {
    this.figurePath = figurePath;
    this.scriptPath = scriptPath;
    this.log = log;
    this.process = undefined;
    this.stopping = false;
    this.url = undefined;
    /** @type {Promise<string>} */
    this.ready = this._start();
  }

  /** @returns {Promise<string>} */
  _start() {
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(
      vscode.Uri.file(this.figurePath)
    )?.uri.fsPath;
    const python = resolvePythonCommand(workspaceFolder);
    const extraArgs = vscode.workspace
      .getConfiguration("figViewer")
      .get("extraPythonArgs", []);

    const args = [
      ...python.args,
      ...extraArgs,
      this.scriptPath,
      this.figurePath,
    ];

    this.log.appendLine(`[fig-viewer] ${python.command} ${args.join(" ")}`);
    const child = spawn(python.command, args, { windowsHide: true });
    this.process = child;

    return new Promise((resolve, reject) => {
      let settled = false;
      const stderrTail = [];

      const rl = readline.createInterface({ input: child.stdout });
      rl.on("line", (line) => {
        this.log.appendLine(`[fig-server] ${line}`);
        if (!settled && line.startsWith(READY_PREFIX)) {
          settled = true;
          this.url = line.slice(READY_PREFIX.length).trim();
          resolve(this.url);
        }
      });
      this.rl = rl;

      child.stderr.on("data", (data) => {
        const text = data.toString();
        this.log.append(`[fig-server] ${text}`);
        stderrTail.push(text);
        if (stderrTail.length > 40) {
          stderrTail.shift();
        }
      });

      child.on("error", (err) => {
        if (!settled) {
          settled = true;
          reject(
            new Error(
              `Failed to launch Python (${python.command}): ${err.message}`
            )
          );
        }
      });

      child.on("exit", (code, signal) => {
        if (this.stopping) {
          return;
        }
        const reason = signal ? `signal ${signal}` : `code ${code}`;
        if (!settled) {
          settled = true;
          reject(
            new Error(
              `Python figure server exited before it was ready (${reason}).\n` +
                stderrTail.join("")
            )
          );
        } else {
          this.log.appendLine(`[fig-server] exited (${reason})`);
        }
      });
    });
  }

  /** Stop the child process. Safe to call multiple times. */
  stop() {
    if (this.stopping) {
      return;
    }
    this.stopping = true;
    if (this.rl) {
      this.rl.close();
      this.rl = undefined;
    }
    const child = this.process;
    if (!child || child.killed) {
      return;
    }
    child.kill();
    const timer = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
      }
    }, 2000);
    child.once("exit", () => clearTimeout(timer));
  }
}

module.exports = { FigServer };
