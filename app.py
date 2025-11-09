# app.py
from flask import Flask, render_template_string, redirect, url_for, jsonify, send_file
import threading, time, random, pickle, os, json, tempfile
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    KOLKATA = ZoneInfo("Asia/Kolkata")
except Exception:
    KOLKATA = None

app = Flask(__name__)

# -------------------------
# Config / Modules
# -------------------------
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "events.log"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODULES = {
    "Heart Rate": {"icon": "❤️", "unit": "bpm", "interval": 5, "file": os.path.join(CHECKPOINT_DIR, "heart.pkl")},
    "Temperature": {"icon": "🌡️", "unit": "°C", "interval": 8, "file": os.path.join(CHECKPOINT_DIR, "temp.pkl")},
    "Oxygen": {"icon": "🫁", "unit": "%", "interval": 10, "file": os.path.join(CHECKPOINT_DIR, "oxy.pkl")},
}

# runtime state for each module
state = {
    name: {
        "value": None,
        "status": "Active",
        "last_cp": None,                      # human readable timestamp
        "history": [],
        "failed": False,
        "last_checkpoint_time_epoch": 0.0,    # numeric used for scheduling
        "seq": 0,                             # sequence number for samples
        "last_value_time": None,              # human readable timestamp for last value update
    }
    for name in MODULES
}

state_lock = threading.Lock()
log_lock = threading.Lock()

# -------------------------
# Helpers
# -------------------------
def now_local_iso():
    if KOLKATA:
        return datetime.now(KOLKATA).strftime("%Y-%m-%d %H:%M:%S %Z")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_log(level, msg):
    entry = {"time": now_local_iso(), "level": level, "msg": msg}
    line = json.dumps(entry, ensure_ascii=False)
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def read_recent_logs(limit=80):
    """Read recent logs safely — skip malformed or incomplete lines."""
    if not os.path.exists(LOG_FILE):
        return []
    entries = []
    with log_lock:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "time" in data and "level" in data and "msg" in data:
                        entries.append(data)
                except json.JSONDecodeError:
                    # Skip any bad lines silently
                    continue
    return list(reversed(entries))[:limit]

# -------------------------
# Checkpointing / Recovery
# -------------------------
def _atomic_write_pickle(path, payload):
    """Write pickle atomically: write to tmp then replace."""
    dir_ = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt_", dir=dir_)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp_path, path)  # atomic on same filesystem
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def save_checkpoint_file(module_name, value):
    meta = MODULES[module_name]
    path = meta["file"]
    try:
        payload = {
            "module": module_name,
            "value": value,
            "unit": meta["unit"],
            "time": now_local_iso(),
            "epoch": time.time(),
        }
        _atomic_write_pickle(path, payload)
        with state_lock:
            state[module_name]["last_cp"] = payload["time"]
            state[module_name]["last_checkpoint_time_epoch"] = payload["epoch"]
        add_log("SUCCESS", f"{module_name}: checkpoint saved @ {state[module_name]['last_cp']}")
    except Exception as e:
        add_log("ERROR", f"{module_name}: checkpoint save failed ({e})")
        with state_lock:
            state[module_name]["status"] = "Checkpoint Error"

def async_save_checkpoint(module_name, value):
    def _job():
        save_checkpoint_file(module_name, value)
        with state_lock:
            if not state[module_name]["failed"]:
                state[module_name]["status"] = "Active"
    threading.Thread(target=_job, daemon=True).start()

def load_checkpoint_file(module_name):
    path = MODULES[module_name]["file"]
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            add_log("ERROR", f"{module_name}: checkpoint load failed ({e})")
    return None

def recover_module(module_name):
    data = load_checkpoint_file(module_name)
    with state_lock:
        if data:
            state[module_name]["value"] = data.get("value")
            state[module_name]["status"] = "Recovered"
            state[module_name]["failed"] = False
            state[module_name]["last_cp"] = data.get("time")
            # do not update last_checkpoint_time_epoch here; it reflects the file's own epoch
            add_log("SUCCESS", f"{module_name}: recovered from checkpoint @ {data.get('time')}")
        else:
            state[module_name]["status"] = "No Checkpoint"
            state[module_name]["failed"] = False
            add_log("WARN", f"{module_name}: no checkpoint available for recovery")

# -------------------------
# Simulation Threads
# -------------------------
def generate_value_for(name):
    if name == "Heart Rate":
        return random.randint(60, 100)
    if name == "Temperature":
        return round(random.uniform(36.0, 37.5), 1)
    if name == "Oxygen":
        return random.randint(92, 100)
    return None

def module_loop(name):
    meta = MODULES[name]
    interval = meta["interval"]
    last_ck = time.time() - (interval // 2)
    with state_lock:
        state[name]["last_checkpoint_time_epoch"] = last_ck

    while True:
        with state_lock:
            failed = state[name]["failed"]

        if failed:
            with state_lock:
                state[name]["status"] = "Recovering..."
            recover_module(name)
            time.sleep(1)
            continue

        new_val = generate_value_for(name)
        with state_lock:
            state[name]["seq"] += 1
            state[name]["value"] = new_val
            state[name]["last_value_time"] = now_local_iso()
            state[name]["history"].append(new_val)
            if len(state[name]["history"]) > 300:
                state[name]["history"] = state[name]["history"][-300:]

        now_epoch = time.time()
        with state_lock:
            last_cp = state[name]["last_checkpoint_time_epoch"]

        if now_epoch - last_cp >= interval:
            with state_lock:
                state[name]["status"] = "Checkpointing..."
            async_save_checkpoint(name, new_val)

        # 1% simulated failure
        if random.random() < 0.01:
            with state_lock:
                state[name]["failed"] = True
                state[name]["status"] = "Failed"
            add_log("ERROR", f"{name}: spontaneous failure (simulated)")

        time.sleep(1)

# -------------------------
# Routes (UI preserved exactly)
# -------------------------
@app.route("/")
def dashboard():
    cards_html = ""
    with state_lock:
        for name, meta in MODULES.items():
            s = state[name]
            display_val = f"{s['value']} {meta['unit']}" if s['value'] is not None else "—"
            status = s['status']
            last_cp = s['last_cp'] or "—"

            if "Active" in status:
                pill = "<span style='background:#dcfce7;color:#16a34a;padding:6px 10px;border-radius:999px;font-weight:700;'>ACTIVE</span>"
            elif "Checkpoint" in status:
                pill = "<span style='background:#fef9c3;color:#f59e0b;padding:6px 10px;border-radius:999px;font-weight:700;'>CHECKPOINTING</span>"
            elif "Failed" in status:
                pill = "<span style='background:#fee2e2;color:#dc2626;padding:6px 10px;border-radius:999px;font-weight:700;'>FAILED</span>"
            elif "Recovered" in status:
                pill = "<span style='background:#dbeafe;color:#2563eb;padding:6px 10px;border-radius:999px;font-weight:700;'>RECOVERED</span>"
            else:
                pill = f"<span style='background:#f1f5f9;color:#64748b;padding:6px 10px;border-radius:999px;font-weight:700;'> {status} </span>"

            cards_html += f"""
            <div style='flex:1; min-width:220px; margin:8px; padding:18px; border-radius:12px; background:white; box-shadow:0 6px 18px rgba(2,6,23,0.06);'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='font-weight:700; font-size:16px;'>{name}</div>
                    <div style='font-size:22px;'>{meta['icon']}</div>
                </div>
                <div style='margin-top:12px; font-weight:800; font-size:28px;'>{display_val}</div>
                <div style='margin-top:10px;'>{pill}</div>
                <div style='margin-top:8px; color:#64748b; font-size:13px;'>Last checkpoint: <strong>{last_cp}</strong></div>
                <div style='margin-top:10px;'>
                    <a href="/fail/{name}" style="color:#dc3545; font-weight:700; text-decoration:none;">💥 Fail</a>
                    &nbsp;&nbsp;
                    <a href="/recover/{name}" style="color:#0d6efd; font-weight:700; text-decoration:none;">🔁 Recover</a>
                </div>
            </div>
            """

    logs = read_recent_logs(80)
    rows = ""
    for entry in logs:
        color = {"SUCCESS": "#16a34a", "ERROR": "#dc2626", "WARN": "#f59e0b"}.get(entry["level"], "#3b82f6")
        rows += f"""
        <tr>
            <td style="padding:10px; color:#64748b; width:220px;">{entry['time']}</td>
            <td style="padding:10px; width:120px;"><span style="color:{color}; font-weight:700;">{entry['level']}</span></td>
            <td style="padding:10px;">{entry['msg']}</td>
        </tr>
        """

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <title>Cloud Health Monitor</title>
        <style>
          body {{ font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial; background:#f1f6fb; margin:0; }}
          .container {{ max-width:1100px; margin:18px auto; padding:18px; }}
          .grid {{ display:flex; gap:12px; flex-wrap:wrap; }}
          .log-table {{ width:100%; border-collapse:collapse; background:white; border-radius:10px; overflow:hidden; }}
        </style>
      </head>
      <body>
        <div class="container">
          <header style="display:flex; justify-content:space-between; align-items:center; padding:12px 0;">
            <div style="display:flex; align-items:center; gap:12px;">
              <div style="font-size:36px;">🏥</div>
              <div>
                <div style="font-weight:800; font-size:20px;">Cloud Health Monitor</div>
                <div style="color:#64748b; font-size:13px;">Asynchronous checkpointing — each module saves independently</div>
              </div>
            </div>
            <div style="display:flex; gap:12px; align-items:center;">
              <a href="/" style="text-decoration:none; color:#0f172a; font-weight:700;">Refresh</a>
              <a href="/reset" style="text-decoration:none; color:#0f172a; font-weight:700;">Reset</a>
              <a href="/api/state" style="text-decoration:none; color:#0f172a; font-weight:700;">API</a>
              <a href="/logs" style="text-decoration:none; color:#0f172a; font-weight:700;">Logs</a>
            </div>
          </header>

          <main>
            <section style="margin-top:18px;">
              <div class="grid">{cards_html}</div>
            </section>

            <section style="margin-top:20px;">
              <h3 style="margin:0 0 8px 0;">Real-Time Event Log</h3>
              <div style="overflow:auto;">
                <table class="log-table" border="0">
                  <thead style="background:#f8fafc;">
                    <tr>
                      <th style="text-align:left; padding:12px; color:#64748b;">Timestamp</th>
                      <th style="text-align:left; padding:12px; color:#64748b;">Level</th>
                      <th style="text-align:left; padding:12px; color:#64748b;">Event</th>
                    </tr>
                  </thead>
                  <tbody>{rows}</tbody>
                </table>
              </div>
            </section>
          </main>
        </div>
        <script>setTimeout(() => window.location.reload(), 2200);</script>
      </body>
    </html>
    """
    return render_template_string(html)

@app.route("/fail/<module_name>")
def fail_module(module_name):
    if module_name in state:
        with state_lock:
            state[module_name]["failed"] = True
            state[module_name]["status"] = "Failed"
        add_log("ERROR", f"{module_name}: manual failure triggered")
    return redirect(url_for("dashboard"))

@app.route("/recover/<module_name>")
def recover_route(module_name):
    if module_name in state:
        with state_lock:
            state[module_name]["failed"] = False
            state[module_name]["status"] = "Recovering..."
        recover_module(module_name)
    return redirect(url_for("dashboard"))

@app.route("/reset")
def reset_all():
    # Clear checkpoints and logs
    for meta in MODULES.values():
        if os.path.exists(meta["file"]):
            try:
                os.remove(meta["file"])
            except Exception:
                pass
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except Exception:
            pass

    with state_lock:
        for name in state:
            state[name] = {
                "value": None, "status": "Active", "last_cp": None, "history": [],
                "failed": False, "last_checkpoint_time_epoch": 0.0, "seq": 0, "last_value_time": None
            }
    add_log("INFO", "System reset: checkpoints and logs cleared")
    return redirect(url_for("dashboard"))

# -------------------------
# Extra helpful endpoints (do not affect UI styling)
# -------------------------
@app.route("/api/state")
def api_state():
    with state_lock:
        snapshot = {
            name: {
                "value": s["value"],
                "unit": MODULES[name]["unit"],
                "status": s["status"],
                "last_checkpoint": s["last_cp"],
                "seq": s["seq"],
                "last_value_time": s["last_value_time"],
                "history_len": len(s["history"]),
                "checkpoint_file": MODULES[name]["file"],
            }
            for name, s in state.items()
        }
    return jsonify({"time": now_local_iso(), "modules": snapshot})

@app.route("/logs")
def download_logs():
    # If log file doesn't exist yet, create an empty one so route still works
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    return send_file(LOG_FILE, as_attachment=True, download_name="events.log", mimetype="text/plain")

# -------------------------
# Start background threads
# -------------------------
for module_name in MODULES:
    threading.Thread(target=module_loop, args=(module_name,), daemon=True).start()

if __name__ == "__main__":
    add_log("INFO", "System starting (Kolkata time).")
    app.run(debug=True, use_reloader=False)
