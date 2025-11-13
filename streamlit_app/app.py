# app.py (Option 2: No environment filter dropdown)
import streamlit as st
import requests
import pandas as pd
import re
import os
import logging
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import matplotlib.pyplot as plt

# ---------------- BASIC LOGGING (prints to terminal) ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
REPO = "ai-autonomous/dqn-project"
PER_PAGE = 10
TIMEZONE_OFFSET = timedelta(hours=5, minutes=30)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

st.set_page_config(page_title="GitHub Actions Dashboard", layout="wide")
st.title("🤖 GitHub Actions Dashboard – ai-autonomous/dqn-project")

# ---------------- AUTH CHECK ----------------
if not GITHUB_TOKEN:
    st.error("⚠️ Missing GitHub Token. Please set GITHUB_TOKEN as an environment variable.")
    st.stop()

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# ---------------- CACHED HTTP HELPERS ----------------
@st.cache_data(ttl=60)
def get_runs_page(url):
    """Return JSON for runs page (cached)."""
    logger.info(f"GET runs url: {url}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def get_jobs_for_run(jobs_url):
    """Return jobs JSON for a run (cached)."""
    logger.info(f"GET jobs url: {jobs_url}")
    r = requests.get(jobs_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def download_job_logs(job_id):
    """Return dict with status_code and bytes content for job logs (cached)."""
    url = f"https://api.github.com/repos/{REPO}/actions/jobs/{job_id}/logs"
    logger.info(f"Downloading logs for job {job_id} from {url}")
    r = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=60)
    # don't call raise_for_status because logs might not be accessible for some jobs
    return {"status_code": r.status_code, "content": r.content}

# ---------------- LOG PARSING ----------------
def parse_logs_for_inputs_and_reward(job_id, raw_log_bytes):
    """
    raw_log_bytes: bytes content of the logs (or b'' if none)
    returns dict with algorithm, environment, Mean Reward
    """
    data = {"algorithm": "N/A", "environment": "N/A", "Mean Reward": "N/A"}

    if not raw_log_bytes:
        logger.info(f"[JOB {job_id}] No raw log bytes.")
        return data

    try:
        text = raw_log_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"[JOB {job_id}] decode error: {e}")
        text = ""

    # Debug snippet to terminal
    logger.debug(f"[JOB {job_id}] log snippet:\n{text[:800]}")

    # Extract inputs block ONLY (workflow must print it)
    block_match = re.search(
        r"=== WORKFLOW INPUTS START ===(.*?)=== WORKFLOW INPUTS END ===",
        text,
        re.S,
    )
    if block_match:
        block = block_match.group(1)
        pairs = re.findall(
            r"(algorithm|environment|total_steps|stage_size|learning_rate):\s*([A-Za-z0-9_\-\.]+)",
            block,
        )
        for k, v in pairs:
            data[k] = v.strip()

    # Extract Mean Reward
    reward_match = re.search(r"Mean Reward:\s*([0-9\.\+\-\s±]+)", text)
    if reward_match:
        data["Mean Reward"] = reward_match.group(1).strip()

    return data

# ---------------- PAGINATION ----------------
if "page" not in st.session_state:
    st.session_state.page = 1

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅️ Previous", disabled=(st.session_state.page == 1)):
        st.session_state.page -= 1
with col3:
    if st.button("Next ➡️"):
        st.session_state.page += 1

page = st.session_state.page
runs_url = f"https://api.github.com/repos/{REPO}/actions/runs?per_page={PER_PAGE}&page={page}"

# ---------------- FETCH WORKFLOW RUNS ----------------
try:
    runs_json = get_runs_page(runs_url)
except Exception as e:
    logger.exception("Failed to fetch workflow runs")
    st.error(f"GitHub API Error fetching runs: {e}")
    st.stop()

runs = runs_json.get("workflow_runs", [])
if not runs:
    st.warning("No workflow runs found.")
    st.stop()

# ---------------- PARSE RUNS -> ROWS ----------------
rows = []
for run in runs:
    # created datetime
    created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) + TIMEZONE_OFFSET
    created_str = created.strftime("%b %d, %I:%M %p")

    jobs_url = run["jobs_url"]
    try:
        jobs_json = get_jobs_for_run(jobs_url)
    except Exception as e:
        logger.warning(f"Skipping run {run.get('id')} — failed to fetch jobs: {e}")
        continue

    for job in jobs_json.get("jobs", []):
        # compute duration
        try:
            start = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
        except Exception:
            start = created
        try:
            end = datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00")) if job.get("completed_at") else start
        except Exception:
            end = start
        dur_sec = int((end - start).total_seconds())
        m, s = divmod(max(dur_sec, 0), 60)

        # download logs (cached)
        log_fetch = download_job_logs(job["id"])
        raw_bytes = log_fetch.get("content", b"") if log_fetch else b""

        # parse logs for inputs & reward
        log_data = parse_logs_for_inputs_and_reward(job["id"], raw_bytes)

        rows.append({
            "algorithm": log_data.get("algorithm", "N/A"),
            "environment": log_data.get("environment", "N/A"),
            "Mean Reward": log_data.get("Mean Reward", "N/A"),
            "Result": job.get("conclusion") or "In Progress",
            "Triggered By": run.get("triggering_actor", {}).get("login", "N/A"),
            "Created At": created_str,
            "Duration": f"{m}m {s}s",
            "Created_At_DT": created,
            "View on GitHub": f"https://github.com/{REPO}/actions/runs/{run['id']}/job/{job['id']}",
        })

# ---------------- DATAFRAME ----------------
df = pd.DataFrame(rows)

# If df is empty, inform and stop
if df.empty:
    st.warning("No job data available (logs may be missing or not contain input block).")
    st.stop()

# ---------------- FILTER BAR (NO ENV FILTER) ----------------
st.subheader("🔍 Filters (Environment filter removed)")

colA, colB = st.columns(2)
with colA:
    algo_values = sorted([a for a in df["algorithm"].unique() if a and a != "N/A"])
    algo_filter = st.selectbox("Filter by Algorithm", ["All"] + algo_values)
with colB:
    result_values = sorted(df["Result"].unique())
    result_filter = st.selectbox("Filter by Result", ["All"] + result_values)

filtered_df = df.copy()
if algo_filter != "All":
    filtered_df = filtered_df[filtered_df["algorithm"] == algo_filter]
if result_filter != "All":
    filtered_df = filtered_df[filtered_df["Result"] == result_filter]

# ---------------- TABLES FIRST ----------------
st.subheader("📋 Job Tables by Environment (no global environment filter)")

env_tables = {
    "lunarlander": filtered_df[filtered_df["environment"] == "lunarlander"],
    "cartpole": filtered_df[filtered_df["environment"] == "cartpole"],
    "recsim": filtered_df[filtered_df["environment"] == "recsim"],
}

def render_table(df_table):
    """Render AG-Grid safely; show an empty Streamlit dataframe if empty."""
    if df_table.empty:
        st.info("No runs available for this environment yet.")
        empty_df = pd.DataFrame(columns=[
            "algorithm", "environment", "Mean Reward",
            "Result", "Created At", "Duration"
        ])
        st.dataframe(empty_df, width='stretch')
        return

    gb = GridOptionsBuilder.from_dataframe(df_table)
    gb.configure_pagination(enabled=True, paginationPageSize=8)
    gb.configure_default_column(editable=False, filter=True, sortable=True, resizable=True)
    gb.configure_column(
        "View on GitHub",
        header_name="View",
        cellRenderer=('function(p){return `<a href="${p.value}" target="_blank">🔗 Open</a>`;}')
    )
    grid_options = gb.build()

    AgGrid(
        df_table,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        height=300,
    )

tab1, tab2, tab3 = st.tabs(["🚀 LunarLander", "🟦 CartPole", "🟣 RecSim"])
with tab1:
    render_table(env_tables["lunarlander"])
with tab2:
    render_table(env_tables["cartpole"])
with tab3:
    render_table(env_tables["recsim"])

# ---------------- CHARTS AFTER TABLES ----------------
st.subheader("📈 Charts (based on filtered data)")

if not filtered_df.empty:
    # Mean Reward Over Time
    st.write("### Mean Reward Over Time")
    fig, ax = plt.subplots()
    cdf = filtered_df.copy()
    cdf["Mean Reward"] = pd.to_numeric(cdf["Mean Reward"], errors="coerce")
    cdf = cdf.dropna(subset=["Mean Reward"])
    if not cdf.empty:
        # sort by datetime to get sensible lines
        cdf = cdf.sort_values("Created_At_DT")
        ax.plot(cdf["Created_At_DT"], cdf["Mean Reward"], marker="o")
        ax.set_xlabel("Created At")
        ax.set_ylabel("Mean Reward")
    st.pyplot(fig)

    # Mean Reward by Algorithm
    st.write("### Mean Reward by Algorithm")
    fig, ax = plt.subplots()
    adf = filtered_df.copy()
    adf["Mean Reward"] = pd.to_numeric(adf["Mean Reward"], errors="coerce")
    grp = adf.groupby("algorithm")["Mean Reward"].mean().dropna()
    if not grp.empty:
        grp.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean Reward (mean)")
    st.pyplot(fig)

    # Mean Reward by Environment
    st.write("### Mean Reward by Environment")
    fig, ax = plt.subplots()
    edf = filtered_df.copy()
    edf["Mean Reward"] = pd.to_numeric(edf["Mean Reward"], errors="coerce")
    g2 = edf.groupby("environment")["Mean Reward"].mean().dropna()
    if not g2.empty:
        g2.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean Reward (mean)")
    st.pyplot(fig)
else:
    st.info("No data available for charts (filtered_df is empty).")

# ---------------- SUMMARY ----------------
st.subheader("📊 Summary Insights")
st.write(f"Algorithms Used: {sorted([a for a in df['algorithm'].unique() if a and a != 'N/A'])}")
st.write(f"Environments: {sorted([e for e in df['environment'].unique() if e and e != 'N/A'])}")
st.write(f"Successful: {int((df['Result'].str.lower() == 'success').sum())}")
st.write(f"Failed: {int((df['Result'].str.lower() == 'failure').sum())}")
st.write(f"In Progress: {int((df['Result'].str.lower() == 'in progress').sum())}")

st.caption("Data fetched from GitHub API — logs parsed for workflow inputs + reward summary.")

