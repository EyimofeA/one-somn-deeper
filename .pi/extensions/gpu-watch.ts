// gpu-watch.ts — project extension: monitor GPU box status and warn on stale state
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const GPU_STATE = "solving/experiments/.gpu_box.json";

export default function (pi: ExtensionAPI) {
  pi.on("session_start", (_event, ctx) => {
    const statePath = join(ctx.cwd, GPU_STATE);
    if (!existsSync(statePath)) return;

    try {
      const data = JSON.parse(readFileSync(statePath, "utf8"));
      const updated = data.updated_at ? new Date(data.updated_at) : null;
      const now = new Date();
      const hoursStale = updated ? (now.getTime() - updated.getTime()) / 36e5 : Infinity;

      if (hoursStale > 24) {
        ctx.ui.notify(
          `GPU state stale (${Math.round(hoursStale)}h). Box may be down — run ./scripts/osmn gpu status`,
          "warning"
        );
      } else if (data.status === "active") {
        const alias = data.alias || data.gpu || "GPU";
        ctx.ui.notify(`${alias} active · ${data.gpu || "unknown"}`, "info");
      }
    } catch {
      // corrupted state file — ignore
    }
  });
}