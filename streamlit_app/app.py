import streamlit as st
import requests
import pandas as pd
import re
import os
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import matplotlib.pyplot as plt

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

# ---------------- HELPER: Parse logs ----------------
def fetch_job_logs_and_summary(repo, job_id, headers):

    url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
    r = requests.get(url, headers=headers, allow_redirects=True)

    data = {
        "algorithm": "N/A",
        "environment": "N/A",
        "Mean Reward": "N/A",
    }

    if r.status_code != 200:
        return data

    log_file = f"job_{job_id}.txt"
    with open(log_file, "wb") as f:
        f.write(r.content)

    try:
        text = open(log_file, "r", errors="ignore").read()

        # Extract inputs only inside workflow inputs block
        block = re.search(
            r"=== WORKFLOW INPUTS START ===(.*?)=== WORKFLOW INPUTS END ===",
            text,
            re.S,
        )
        if block:
            pairs = re.findall(
                r"(algorithm|environment):\s*([A-Za-z0-9_\-\.]+)",
                block.group(1),
            )
            for k, v in pairs:
                data[k] = v.strip()

        # Extract Mean Reward
        match = re.search(r"Mean Reward:\s*([0-9\.\+\-\s±]+)", text)
        if match:
            data["Mean Reward"] = match.group(1).strip()

    finally:
        if os.path.exists(log_file):
            os.remove(log_file)

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
url = f"https://api.github.com/repos/{REPO}/actions/runs?per_page={PER_PAGE}&page={page}"

# ---------------- FETCH WORKFLOW RUNS ----------------
resp = requests.get(url, headers=HEADERS)
if resp.status_code != 200:
    st.error(f"GitHub API Error {resp.status_code}")
    st.stop()

runs = resp.json().get("workflow_runs", [])
if not runs:
    st.warning("No workflow runs found.")
    st.stop()

# ---------------- PARSE RUNS ----------------
rows = []
for run in runs:

    created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) + TIMEZONE_OFFSET
    created_str = created.strftime("%b %d, %I:%M %p")

    jobs_url = run["jobs_url"]
    jobs_resp = requests.get(jobs_url, headers=HEADERS)
    if jobs_resp.status_code != 200:
        continue

    for job in jobs_resp.json().get("jobs", []):

        start = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00")) if job.get("completed_at") else start
        dur_sec = int((end - start).total_seconds())
        m, s = divmod(dur_sec, 60)

        log_data = fetch_job_logs_and_summary(REPO, job["id"], HEADERS)

        rows.append({
            "algorithm": log_data["algorithm"],
            "environment": log_data["environment"],
            "Mean Reward": log_data["Mean Reward"],
            "Result": job["conclusion"] or "In Progress",
            "Triggered By": run.get("triggering_actor", {}).get("login", "N/A"),
            "Created At": created_str,
            "Duration": f"{m}m {s}s",
            "Created_At_DT": created,
            "View on GitHub": f"https://github.com/{REPO}/actions/runs/{run['id']}/job/{job['id']}",
        })

# ---------------- DATAFRAME ----------------
df = pd.DataFrame(rows)
st.success(f"Showing {len(df)} job runs on page {page}")

# ---------------- FILTER BAR ----------------
st.subheader("🔍 Filters")

colA, colB, colC = st.columns(3)
with colA:
    algo_filter = st.selectbox("Filter by Algorithm", ["All"] + sorted(df["algorithm"].unique()))
with colB:
    env_filter = st.selectbox("Filter by Environment", ["All"] + sorted(df["environment"].unique()))
with colC:
    result_filter = st.selectbox("Filter by Result", ["All"] + sorted(df["Result"].unique()))

# Apply filters
filtered_df = df.copy()

if algo_filter != "All":
    filtered_df = filtered_df[filtered_df["algorithm"] == algo_filter]

if env_filter != "All":
    filtered_df = filtered_df[filtered_df["environment"] == env_filter]

if result_filter != "All":
    filtered_df = filtered_df[filtered_df["Result"] == result_filter]


# ---------------- TABLES FIRST ----------------
st.subheader("📋 Job Tables by Environment")

env_tables = {
    "lunarlander": filtered_df[filtered_df["environment"] == "lunarlander"],
    "cartpole": filtered_df[filtered_df["environment"] == "cartpole"],
    "recsim": filtered_df[filtered_df["environment"] == "recsim"],
}

# ---- TABLE RENDERER WITH EMPTY SAFE MODE ----
def render_table(df_table):

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


# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🚀 LunarLander", "🟦 CartPole", "🟣 RecSim"])

with tab1:
    render_table(env_tables["lunarlander"])

with tab2:
    render_table(env_tables["cartpole"])

with tab3:
    render_table(env_tables["recsim"])


# ---------------- CHARTS AFTER TABLES ----------------
st.subheader("📈 Charts")

if not filtered_df.empty:

    # --- Chart: Mean Reward Over Time ---
    st.write("### Mean Reward Over Time")
    fig, ax = plt.subplots()
    cdf = filtered_df.copy()
    cdf["Mean Reward"] = pd.to_numeric(cdf["Mean Reward"], errors="coerce")
    cdf = cdf.dropna(subset=["Mean Reward"])
    if not cdf.empty:
        ax.plot(cdf["Created_At_DT"], cdf["Mean Reward"])
    st.pyplot(fig)

    # --- Chart: Mean Reward by Algorithm ---
    st.write("### Mean Reward by Algorithm")
    fig, ax = plt.subplots()
    adf = filtered_df.copy()
    adf["Mean Reward"] = pd.to_numeric(adf["Mean Reward"], errors="coerce")
    adf.groupby("algorithm")["Mean Reward"].mean().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # --- Chart: Mean Reward by Environment ---
    st.write("### Mean Reward by Environment")
    fig, ax = plt.subplots()
    edf = filtered_df.copy()
    edf["Mean Reward"] = pd.to_numeric(edf["Mean Reward"], errors="coerce")
    edf.groupby("environment")["Mean Reward"].mean().plot(kind="bar", ax=ax)
    st.pyplot(fig)

else:
    st.info("No data available for charts.")


# ---------------- SUMMARY ----------------
st.subheader("📊 Summary Insights")

st.write(f"Algorithms Used: {sorted(df['algorithm'].unique())}")
st.write(f"Environments: {sorted(df['environment'].unique())}")
st.write(f"Successful: {sum(df['Result'].str.lower() == 'success')}")
st.write(f"Failed: {sum(df['Result'].str.lower() == 'failure')}")
st.write(f"In Progress: {sum(df['Result'].str.lower() == 'in progress')}")

st.caption("Data fetched from GitHub API — logs parsed for workflow inputs + reward summary.")
