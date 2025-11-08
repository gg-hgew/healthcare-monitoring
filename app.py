import streamlit as st
import threading
import time
import random
import pickle
import os
from datetime import datetime

# ----------------------
# Config
# ----------------------
st.set_page_config(page_title="Healthcare Async Checkpointing", layout="wide")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODULES = {
    "Heart Rate": {"icon": "❤️", "unit": "bpm", "file": os.path.join(CHECKPOINT_DIR, "heart.pkl"), "interval": 5},
    "Temperature": {"icon": "🌡️", "unit": "°C", "file": os.path.join(CHECKPOINT_DIR, "temp.pkl"), "interval": 8},
    "Oxygen": {"icon": "🩸", "unit": "%", "file": os.path.join(CHECKPOINT_DIR, "oxy.pkl"), "interval": 10},
}

# ----------------------
# Helpers
# ----------------------

def now_str():
    return datetime.now().strftime("%H:%M:%S")


def generate_value(name):
    if name == "Heart Rate":
        return random.randint(60, 100)
    if name == "Temperature":
        return round(random.uniform(36.0, 37.5), 1)
    if name == "Oxygen":
        return random.randint(92, 100)


# Use a non-blocking save: spawn a thread so UI stays responsive
def async_save_checkpoint(name, value):
    def _save():
        path = MODULES[name]["file"]
        try:
            with open(path, "wb") as f:
                pickle.dump({"value": value, "time": now_str()}, f)
            add_log(f"{MODULES[name]['icon']} {name}: Checkpoint saved @ {now_str()}")
            st.session_state.modules[name]["last_cp"] = now_str()
            st.session_state.modules[name]["status"] = "🟡 Checkpoint saved"
        except Exception as e:
            add_log(f"⚠️ {name}: Checkpoint failed ({e})")
            st.session_state.modules[name]["status"] = "⚠️ Checkpoint error"

    t = threading.Thread(target=_save, daemon=True)
    t.start()


def recover_from_checkpoint(name):
    path = MODULES[name]["file"]
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            st.session_state.modules[name]["value"] = data.get("value")
            st.session_state.modules[name]["status"] = "✅ Recovered from checkpoint"
            add_log(f"{MODULES[name]['icon']} {name}: Recovered from checkpoint @ {data.get('time')}")
        except Exception as e:
            st.session_state.modules[name]["status"] = "⚠️ Recovery error"
            add_log(f"⚠️ {name}: Recovery failed ({e})")
    else:
        st.session_state.modules[name]["status"] = "⚠️ No checkpoint"
        add_log(f"⚠️ {MODULES[name]['icon']} {name}: No checkpoint found for recovery")

    st.session_state.modules[name]["failed"] = False


def add_log(message):
    timestamped = f"[{now_str()}] {message}"
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.insert(0, timestamped)
    # keep logs reasonable
    st.session_state.logs = st.session_state.logs[:200]


# ----------------------
# Initialize session state
# ----------------------
if "modules" not in st.session_state:
    st.session_state.modules = {
        name: {"value": None, "status": "🟢 Active", "last_cp": None, "failed": False, "history": []}
        for name in MODULES
    }

if "logs" not in st.session_state:
    st.session_state.logs = [f"[{now_str()}] System: Ready"]

# ----------------------
# UI - Header & Controls
# ----------------------
st.title("🏥 Healthcare Monitoring Cloud — Async Checkpointing Demo")
st.markdown(
    "This demo simulates multiple independent monitoring modules. Each module periodically saves an asynchronous checkpoint. "
    "On failure, only the affected module restores from its last checkpoint while others keep running (non-blocking recovery)."
)

col_main, col_logs = st.columns([3, 1])

with col_logs:
    st.subheader("📋 System Logs")
    st.write("(most recent first)")
    st.text_area("logs", value="\n".join(st.session_state.logs[:30]), height=480)

# ----------------------
# Module Cards
# ----------------------
cols = st.columns(len(MODULES))
for i, (name, meta) in enumerate(MODULES.items()):
    with cols[i]:
        card = st.container()
        with card:
            st.markdown(f"### {meta['icon']} {name}")

            state = st.session_state.modules[name]

            # simulate manual failure
            fail_btn = st.button(f"💥 Fail {name}", key=f"fail_{name}")
            if fail_btn:
                state["failed"] = True
                state["status"] = "🔴 Failed (Recovering...)"
                add_log(f"{meta['icon']} {name}: Simulated failure triggered")

            # If failed, attempt recovery immediately (non-blocking)
            if state["failed"]:
                recover_from_checkpoint(name)

            if not state["failed"]:
                new_val = generate_value(name)
                state["value"] = new_val
                state["history"].append(new_val)
                # keep history small for charts
                state["history"] = state["history"][-120:]

                # Checkpointing logic: non-blocking save when the interval boundary is hit
                try:
                    current_seconds = int(time.time())
                    if current_seconds % meta["interval"] == 0:
                        async_save_checkpoint(name, new_val)
                    else:
                        # if not checkpointing, ensure status is active unless recovery message present
                        if "Recovered" not in state["status"] and "Checkpoint" not in state["status"]:
                            state["status"] = "🟢 Active"
                except Exception as e:
                    add_log(f"⚠️ {name}: Error in checkpointing logic ({e})")


                state["value"] = new_val
                state["history"].append(new_val)
                # keep history small for charts
                state["history"] = state["history"][-120:]

                # Checkpointing logic: non-blocking save when the interval boundary is hit
                # We use modulo of seconds so different modules will checkpoint at different times
                try:
                    current_seconds = int(time.time())
                    if current_seconds % meta["interval"] == 0:
                        async_save_checkpoint(name, new_val)
                    else:
                        # if not checkpointing, ensure status is active unless recovery message present
                        if "Recovered" not in state["status"] and "Checkpoint" not in state["status"]:
                            state["status"] = "🟢 Active"
                except Exception as e:
                    add_log(f"⚠️ {name}: Error in checkpointing logic ({e})")

            # Display metrics and status with simple colored badges
            value_display = f"{state['value']} {meta['unit']}" if state['value'] is not None else "—"
            st.metric(label="Current Value", value=value_display)

            # Status badge
            status = state["status"]
            # Simple color mapping
            color = "black"
            if "Active" in status:
                color = "green"
            elif "Checkpoint" in status:
                color = "orange"
            elif "Failed" in status or "error" in status:
                color = "red"
            elif "Recovered" in status:
                color = "blue"

            st.markdown(f"**Status:** <span style='color:{color}; font-weight:600'>{status}</span>", unsafe_allow_html=True)
            st.markdown(f"**Last checkpoint:** {state['last_cp']}")

            # Chart
            if len(state["history"]) > 1:
                st.line_chart(state["history"], height=180)

            # Small spacer
            st.write("\n")

# ----------------------
# Footer / Controls
# ----------------------
st.markdown("---")
ctrl_col1, ctrl_col2 = st.columns([1, 3])
with ctrl_col1:
    if st.button("🧹 Reset Checkpoints & Logs"):
        # remove checkpoint files
        for meta in MODULES.values():
            try:
                if os.path.exists(meta["file"]):
                    os.remove(meta["file"])
            except:
                pass
        st.session_state.logs = [f"[{now_str()}] System: Checkpoints cleared"]
        for name in MODULES:
            st.session_state.modules[name] = {"value": None, "status": "🟢 Active", "last_cp": None, "failed": False, "history": []}
        st.experimental_rerun()

with ctrl_col2:
    st.markdown("**Demo Controls / Notes**")
    st.write("• Click a module's 'Fail' button to simulate a crash. The module will attempt to recover from its last checkpoint immediately.")
    st.write("• Checkpoint saves are non-blocking — they run in background threads so UI remains responsive.")
    st.write("• Use 'Reset Checkpoints & Logs' to clear saved files and start fresh.")

# ----------------------
# Auto refresh loop
# ----------------------
# A short sleep and rerun gives the illusion of a continuously updating dashboard.
# Keep it short but not zero to avoid busy looping.

time.sleep(1)
st.experimental_rerun()
