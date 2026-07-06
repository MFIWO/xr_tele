#!/usr/bin/env python3
"""Browser heatmap host for RH5DG2 tactile UDP JSON packets.

This tool intentionally stays independent from the teleop control loop. It only
borrows the Robotis tactile processing pattern: collect a quiet baseline,
subtract it, clip negative values, apply a deadband, then smooth with EMA.
"""

import argparse
import copy
import json
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from teleop.utils.rh5dg2_tactile import RH5DG2TactileHeatMapper, normalize_packet  # noqa: E402


class TactileState:
    def __init__(self, args):
        self.args = args
        self.mapper = RH5DG2TactileHeatMapper(
            side=args.side,
            baseline_seconds=args.baseline_seconds,
            ema_alpha=args.ema_alpha,
            deadband=args.deadband,
            normal_max=args.normal_max,
            tangent_max=args.tangent_max,
            proximity_max=args.proximity_max,
            proximity_weight=args.proximity_weight,
        )
        self.lock = threading.Lock()
        self.latest_raw = normalize_packet({}, args.side)
        self.latest_view = self._empty_view()
        self.packet_count = 0
        self.error_count = 0
        self.last_rx_time = 0.0
        self.last_error = ""

    def update_packet(self, packet):
        raw = normalize_packet(packet, self.args.side)
        view = self.mapper.update(packet)
        with self.lock:
            self.latest_raw = raw
            self.latest_view = view
            self.packet_count += 1
            self.last_rx_time = time.time()

    def mark_error(self, message):
        with self.lock:
            self.error_count += 1
            self.last_error = message

    def reset_baseline(self):
        self.mapper.reset_baseline()

    def snapshot(self):
        with self.lock:
            age = None if self.last_rx_time <= 0.0 else time.time() - self.last_rx_time
            out = copy.deepcopy(self.latest_view)
            out["stats"] = {
                "packets": self.packet_count,
                "errors": self.error_count,
                "age": age,
                "stale": age is None or age > self.args.stale_after,
                "last_error": self.last_error,
                "baseline": self.mapper.status(),
            }
            return out

    def _empty_view(self):
        return self.mapper.update({})


def udp_loop(args, state, stop_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.udp_host, args.udp_port))
    sock.settimeout(0.2)
    print(f"[rh5dg2_tactile_heatmap] UDP listen {args.udp_host}:{args.udp_port}", flush=True)
    try:
        while not stop_event.is_set():
            try:
                raw, _addr = sock.recvfrom(args.recv_size)
            except socket.timeout:
                continue
            except OSError:
                if not stop_event.is_set():
                    state.mark_error("udp socket closed")
                break
            try:
                packet = json.loads(raw.decode("utf-8"))
                if not isinstance(packet, dict):
                    raise ValueError("packet is not a JSON object")
                state.update_packet(packet)
            except Exception as exc:
                state.mark_error(str(exc))
    finally:
        sock.close()


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RH5DG2 Tactile Heatmap</title>
<style>
:root {
  color-scheme: dark;
  --bg: #101316;
  --panel: #181d22;
  --panel-2: #20272e;
  --text: #ecf1f4;
  --muted: #9aa8b2;
  --line: #34414b;
  --hot: 0;
  --green: #48d597;
  --yellow: #ffd166;
  --red: #ef476f;
  --cyan: #5bc0eb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  display: grid;
  grid-template-columns: minmax(320px, 1.05fr) minmax(320px, 0.95fr);
  gap: 18px;
  width: min(1180px, calc(100vw - 32px));
  margin: 18px auto;
}
header {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 12px;
}
h1 { margin: 0; font-size: 22px; font-weight: 720; letter-spacing: 0; }
.status { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: flex-end; }
.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 6px 10px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1;
  background: #14191e;
}
.pill.ok { color: #0d2018; background: var(--green); border-color: transparent; }
.pill.warn { color: #261a03; background: var(--yellow); border-color: transparent; }
.pill.bad { color: #2a020b; background: var(--red); border-color: transparent; }
section {
  min-width: 0;
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 14px;
}
.hand {
  display: grid;
  grid-template-rows: 1fr auto;
  gap: 14px;
  min-height: 660px;
}
.fingers {
  display: grid;
  grid-template-columns: repeat(5, minmax(58px, 1fr));
  align-items: end;
  gap: 10px;
}
.finger {
  min-height: var(--height);
  border: 1px solid var(--line);
  border-radius: 8px 8px 5px 5px;
  background: linear-gradient(to top, var(--fill) 0%, var(--fill) var(--pct), var(--panel-2) var(--pct), var(--panel-2) 100%);
  display: grid;
  align-items: end;
  padding: 8px 6px;
  transition: background 80ms linear;
}
.finger .name, .palm-cell .name {
  color: var(--text);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  text-align: center;
  overflow-wrap: anywhere;
}
.finger .heat {
  color: var(--muted);
  font-size: 18px;
  font-weight: 760;
  text-align: center;
}
.palm {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
}
.palm-cell {
  min-height: 164px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: linear-gradient(to top, var(--fill) 0%, var(--fill) var(--pct), var(--panel-2) var(--pct), var(--panel-2) 100%);
  display: grid;
  align-content: end;
  gap: 8px;
  padding: 10px;
}
.metrics {
  display: grid;
  gap: 10px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
th, td {
  text-align: right;
  border-bottom: 1px solid var(--line);
  padding: 7px 6px;
  white-space: nowrap;
}
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; }
.bar {
  width: 100%;
  height: 9px;
  border-radius: 999px;
  background: #11161a;
  overflow: hidden;
  border: 1px solid #28333b;
}
.bar > i { display: block; height: 100%; width: var(--pct); background: var(--fill); }
button {
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #242c33;
  color: var(--text);
  padding: 8px 11px;
  font-weight: 700;
  cursor: pointer;
}
button:hover { border-color: var(--cyan); }
.note { color: var(--muted); font-size: 12px; line-height: 1.45; margin: 0; }
@media (max-width: 860px) {
  main { grid-template-columns: 1fr; width: min(100vw - 20px, 680px); }
  header { align-items: flex-start; flex-direction: column; }
  .hand { min-height: 560px; }
  .finger { min-height: calc(var(--height) * .76); }
}
</style>
</head>
<body>
<main>
  <header>
    <h1>RH5DG2 Tactile Heatmap</h1>
    <div class="status">
      <span id="side" class="pill">side</span>
      <span id="rx" class="pill bad">waiting</span>
      <span id="baseline" class="pill warn">baseline</span>
      <span id="packets" class="pill">0 packets</span>
      <button id="reset">Reset Baseline</button>
    </div>
  </header>
  <section class="hand">
    <div id="fingers" class="fingers"></div>
    <div id="palm" class="palm"></div>
  </section>
  <section class="metrics">
    <table>
      <thead>
        <tr><th>finger</th><th>heat</th><th>N raw</th><th>T raw</th><th>dir</th><th>prox raw</th></tr>
      </thead>
      <tbody id="fingerRows"></tbody>
    </table>
    <table>
      <thead>
        <tr><th>palm</th><th>heat</th><th>N raw</th><th>T raw</th><th>dir</th></tr>
      </thead>
      <tbody id="palmRows"></tbody>
    </table>
    <p class="note">Heat uses baseline-corrected normal, tangential, and finger proximity channels. Direction is displayed only when it is 0-359 degrees; 65535 means invalid/no stable tangential direction.</p>
  </section>
</main>
<script>
const fingerHeights = { thumb: "64%", index: "92%", middle: "100%", ring: "88%", little: "70%" };
const fmt = (value, digits = 0) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "-";
const heatColor = (h) => {
  const hue = 198 - Math.max(0, Math.min(1, h)) * 165;
  const light = 45 + Math.max(0, Math.min(1, h)) * 10;
  return `hsl(${hue} 86% ${light}%)`;
};
const setHeatVars = (el, heat) => {
  const pct = `${Math.round(Math.max(0, Math.min(1, heat)) * 100)}%`;
  el.style.setProperty("--pct", pct);
  el.style.setProperty("--fill", heatColor(heat));
};
const dirText = (item) => item.direction_valid ? `${fmt(item.direction_raw)} deg` : "invalid";

function render(data) {
  document.getElementById("side").textContent = data.side || "side";
  const stats = data.stats || {};
  const rx = document.getElementById("rx");
  if (stats.stale) {
    rx.textContent = stats.age == null ? "waiting" : `stale ${fmt(stats.age, 1)}s`;
    rx.className = "pill bad";
  } else {
    rx.textContent = `live ${fmt(stats.age, 2)}s`;
    rx.className = "pill ok";
  }
  const baseline = stats.baseline || {};
  const base = document.getElementById("baseline");
  base.textContent = baseline.ready ? `baseline ${baseline.samples}` : `calibrating ${fmt(baseline.elapsed, 1)}s`;
  base.className = baseline.ready ? "pill ok" : "pill warn";
  document.getElementById("packets").textContent = `${stats.packets || 0} packets`;

  const fingers = document.getElementById("fingers");
  fingers.innerHTML = "";
  for (const item of data.fingers || []) {
    const div = document.createElement("div");
    div.className = "finger";
    div.style.setProperty("--height", fingerHeights[item.name] || "80%");
    setHeatVars(div, item.heat);
    div.innerHTML = `<div><div class="heat">${fmt(item.heat * 100)}</div><div class="name">${item.name}</div></div>`;
    fingers.appendChild(div);
  }

  const palm = document.getElementById("palm");
  palm.innerHTML = "";
  for (const item of data.palm || []) {
    const div = document.createElement("div");
    div.className = "palm-cell";
    setHeatVars(div, item.heat);
    div.innerHTML = `<div class="heat">${fmt(item.heat * 100)}</div><div class="name">palm ${item.name}</div>`;
    palm.appendChild(div);
  }

  const fingerRows = document.getElementById("fingerRows");
  fingerRows.innerHTML = "";
  for (const item of data.fingers || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${item.name}<div class="bar"><i></i></div></td><td>${fmt(item.heat * 100)}</td><td>${fmt(item.normal_raw)}</td><td>${fmt(item.tangent_raw)}</td><td>${dirText(item)}</td><td>${fmt(item.proximity_raw)}</td>`;
    setHeatVars(tr.querySelector(".bar"), item.heat);
    fingerRows.appendChild(tr);
  }

  const palmRows = document.getElementById("palmRows");
  palmRows.innerHTML = "";
  for (const item of data.palm || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${item.name}<div class="bar"><i></i></div></td><td>${fmt(item.heat * 100)}</td><td>${fmt(item.normal_raw)}</td><td>${fmt(item.tangent_raw)}</td><td>${dirText(item)}</td>`;
    setHeatVars(tr.querySelector(".bar"), item.heat);
    palmRows.appendChild(tr);
  }
}

document.getElementById("reset").addEventListener("click", async () => {
  await fetch("/reset_baseline", { method: "POST" });
});

const events = new EventSource("/events");
events.onmessage = (event) => render(JSON.parse(event.data));
events.onerror = () => {
  const rx = document.getElementById("rx");
  rx.textContent = "disconnected";
  rx.className = "pill bad";
};
</script>
</body>
</html>
"""


class HeatmapHandler(BaseHTTPRequestHandler):
    server_version = "RH5DG2TactileHeatmap/1.0"

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/snapshot":
            self._send_json(self.server.state.snapshot())
            return
        if self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            interval = 1.0 / max(self.server.args.update_hz, 1.0)
            while not self.server.stop_event.is_set():
                payload = json.dumps(self.server.state.snapshot(), separators=(",", ":"))
                try:
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(interval)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path == "/reset_baseline":
            self.server.state.reset_baseline()
            self._send_json({"ok": True})
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        return

    def _send_json(self, obj):
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class HeatmapServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, args, state, stop_event):
        super().__init__(address, handler)
        self.args = args
        self.state = state
        self.stop_event = stop_event
        self.timeout = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description="Serve an RH5DG2 tactile UDP heatmap in a browser.")
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=56010)
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8088)
    parser.add_argument("--side", default="right_ee")
    parser.add_argument("--recv-size", type=int, default=8192)
    parser.add_argument("--update-hz", type=float, default=30.0)
    parser.add_argument("--stale-after", type=float, default=1.0)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument("--deadband", type=float, default=1.0)
    parser.add_argument("--normal-max", type=float, default=800.0)
    parser.add_argument("--tangent-max", type=float, default=800.0)
    parser.add_argument("--proximity-max", type=float, default=65535.0)
    parser.add_argument("--proximity-weight", type=float, default=0.65)
    return parser.parse_args()


def main():
    args = parse_args()
    stop_event = threading.Event()
    state = TactileState(args)

    def stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    udp_thread = threading.Thread(target=udp_loop, args=(args, state, stop_event), daemon=True)
    udp_thread.start()

    server = HeatmapServer((args.http_host, args.http_port), HeatmapHandler, args, state, stop_event)
    print(
        f"[rh5dg2_tactile_heatmap] HTTP http://{args.http_host}:{args.http_port} "
        f"side={args.side}",
        flush=True,
    )
    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        stop_event.set()
        server.server_close()
        udp_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
