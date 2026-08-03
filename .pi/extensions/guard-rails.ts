// guard-rails.ts — safety gates: block dangerous commands, protect submissions, warn on cost
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { isToolCallEventType } from "@earendil-works/pi-coding-agent";

const FORBIDDEN_BASH = [
  /rm\s+-rf\s+\/(\s|$)/,       // rm -rf /
  /sudo\s+rm/,                   // sudo rm
  />\s*\/dev\/sd/,              // overwrite disk
  /mkfs\./,                      // format filesystem
  /dd\s+if=.*of=\/dev/,         // raw disk write
  /:\s*\{\s*:\s*\|\s*:\s*&\s*\}/, // fork bomb
  /chmod\s+777\s+\//,           // world-writable root
  /git\s+push\s+--force.*main/, // force push main
  /one-layer\s+submit.*--tier\s+hard/i, // auto hard submit
];

const PROTECTED_FILES = [
  /submission\.py$/,
  /RESEARCH_PROTOCOL\.md$/,
  /AGENTS\.md$/,
  /\.env$/,
  /\.gpu_box\.json$/,
];

export default function (pi: ExtensionAPI) {
  // ---- Block dangerous bash commands ----
  pi.on("tool_call", async (event, ctx) => {
    if (isToolCallEventType("bash", event)) {
      const cmd = event.input.command || "";

      for (const pattern of FORBIDDEN_BASH) {
        if (pattern.test(cmd)) {
          ctx.ui.notify(`Blocked: "${cmd.slice(0, 80)}" matches forbidden pattern ${pattern}`, "error");
          return { block: true, reason: `Command matches forbidden pattern: ${pattern}` };
        }
      }
    }
  });

  // ---- Protect submission.py and key files ----
  pi.on("tool_call", async (event, ctx) => {
    if (event.toolName === "write" || event.toolName === "edit") {
      const path: string = (event.input as any).path || "";
      for (const pattern of PROTECTED_FILES) {
        if (pattern.test(path)) {
          const ok = await ctx.ui.confirm(
            "Protected file",
            `${path} is protected. Edit anyway?`
          );
          if (!ok) return { block: true, reason: "User declined protected file edit" };
          break; // confirmed, allow
        }
      }
    }
  });

  // ---- Warn on expensive model calls (high thinking on cheap tasks) ----
  pi.on("before_agent_start", (_event, ctx) => {
    try {
      const lvl = (ctx as any).thinkingLevel;
      if (lvl === "xhigh" || lvl === "max") {
        ctx.ui.notify(
          `Thinking level: ${lvl} — expensive. Consider lowering for non-research tasks.`,
          "warning"
        );
      }
    } catch {}
  });
}