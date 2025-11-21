# app.py – Enhanced UI Version 🚀

import streamlit as st
import requests
import pandas as pd
import re
import os
import logging
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# ---------------- BASIC LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
REPO = "ai-autonomous/dqn-project"
PER_PAGE = 10  # pagination size
TIMEZONE_OFFSET = timedelta(hours=5, minutes=30)
TOKEN = os.getenv("GITHUB_TOKEN")

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="RL Training Dashboard", layout="wide")
st.markdown(
    """
    <style>
        .main { background: #121212; color: #fff; }
        h1 { color: #00e5ff !important; }
        .stButton>button { border-radius: 8px; padding: 6px 16px; font-weight:600; }
    </style>
""", unsafe_allow_html=True)
st.title("🤖 Deep Reinforcement Learning Dashboard")

# ---------------- AUTH CHECK ----------------
if not TOKEN:
    st.error("⚠️ Missing GitHub Token. Please set GITHUB_TOKEN environment variable.")
    st.stop()

HEADERS = {"Authorization": f"token {TOKEN}"}

# ---------------- CACHED HTTP HELPERS ----------------
@st.cache_data(ttl=60)
def get_runs(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def get_jobs(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def download_logs(job_id):
    url = f"https://api.github.com/repos/{REPO}/actions/jobs/{job_id}/logs"
    r = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=60)
    return r.content

# ---------------- LOG PARSER ----------------
def parse_log(log_bytes):
    d = {"algorithm": "N/A", "environment": "N/A", "Mean Reward": "N/A"}
    if not log_bytes: return d

    try:
        text = log_bytes.decode("utf-8", errors="ignore")
    except:
        return d

    block = re.search(r"=== WORKFLOW INPUTS START ===(.*?)=== WORKFLOW INPUTS END ===", text, re.S)
    if block:
        pairs = re.findall(r"(algorithm|environment):\s*([A-Za-z0-9_\-\.]+)", block.group(1))
        for k, v in pairs: d[k] = v

    reward = re.search(r"Mean Reward:\s*([0-9\.\+\-\s±]+)", text)
    if reward: d["Mean Reward"] = reward.group(1).strip()

    return d

# ---------------- GLOBAL PAGE STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = 1

url = f"https://api.github.com/repos/{REPO}/actions/runs?per_page={PER_PAGE}&page={st.session_state.page}"

runs_json = get_runs(url)
runs = runs_json.get("workflow_runs", [])
if not runs:
    st.warning("No workflow runs available.")
    st.stop()

# ---------------- PROCESS DATA ----------------
rows = []
for run in runs:
    created = datetime.fromisoformat(run["created_at"].replace("Z","+00:00")) + TIMEZONE_OFFSET
    created_str = created.strftime("%b %d, %I:%M %p")

    jobs_json = get_jobs(run["jobs_url"])
    for job in jobs_json.get("jobs", []):
        # duration
        try: s = datetime.fromisoformat(job["started_at"].replace("Z","+00:00"))
        except: s = created
        try: e = datetime.fromisoformat(job["completed_at"].replace("Z","+00:00"))
        except: e = s
        m, s2 = divmod(max(int((e-s).total_seconds()),0),60)

        logs = download_logs(job["id"])
        p = parse_log(logs)

        rows.append({
            "algorithm": p["algorithm"],
            "environment": p["environment"],
            "Mean Reward": p["Mean Reward"],
            "Result": job.get("conclusion") or "In Progress",
            "Created At": created_str,
            "Duration": f"{m}m {s2}s",
            "View": f"https://github.com/{REPO}/actions/runs/{run['id']}/job/{job['id']}",
        })

df = pd.DataFrame(rows)

# ---------------- JS STYLES ----------------
link_renderer = JsCode("""
class LinkRenderer {
  init(params){
    this.eGui = document.createElement('span');
    if (params.value) {
      this.eGui.innerHTML = '<a href="' + params.value + '" target="_blank" ' +
        'style="color:#00e5ff;text-decoration:none;font-weight:600;">🔗 Open</a>';
    }
  }
  getGui(){ return this.eGui; }
}
""")

badge_renderer = JsCode("""
class BadgeRenderer {
  init(params){
    this.eGui = document.createElement('span');
    var v = params.value ? params.value.toLowerCase() : "";

    var style = "padding:3px 7px;border-radius:6px;font-size:12px;color:white;font-weight:700;";

    if (v === "success") {
      this.eGui.innerHTML = '<span style="background:#00c853;' + style + '">SUCCESS</span>';
    } else if (v === "failure") {
      this.eGui.innerHTML = '<span style="background:#d50000;' + style + '">FAILED</span>';
    } else {
      this.eGui.innerHTML = '<span style="background:#ffab00;' + style + '">RUNNING</span>';
    }
  }
  getGui(){ return this.eGui; }
}
""")

# ---------------- TABLE RENDER ----------------
def show_table(env_key):
    st.subheader(f"📌 {env_key.upper()} Runs")
    df_env = df[df["environment"] == env_key]

    gb = GridOptionsBuilder.from_dataframe(df_env)
    gb.configure_default_column(editable=False, sortable=True, filter=True, resizable=True)
    gb.configure_column("Result", cellRenderer=badge_renderer)
    gb.configure_column("View", header_name="🔗 Link", cellRenderer=link_renderer)

    grid = gb.build()

    AgGrid(
        df_env,
        gridOptions=grid,
        enable_enterprise_modules=False,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.NO_UPDATE,
        height=350,
        key=f"grid_{env_key}"
    )

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🚀 LunarLander", "🟦 CartPole", "🟣 RecSim"])
with tab1: show_table("lunarlander")
with tab2: show_table("cartpole")
with tab3: show_table("recsim")

# ---------------- PAGE NAVIGATION ----------------
col1, col2 = st.columns(2)
with col1:
    if st.button("◀️ Prev", disabled=(st.session_state.page==1)):
        st.session_state.page -= 1
        st.rerun()
with col2:
    if st.button("Next ▶️"):
        st.session_state.page += 1
        st.rerun()
