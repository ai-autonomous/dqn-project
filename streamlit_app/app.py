import streamlit as st
import requests
import pandas as pd
import re
import os
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ---------------- CONFIG ----------------
REPO = "ai-autonomous/dqn-project"
PER_PAGE = 10
TIMEZONE_OFFSET = timedelta(hours=5, minutes=30)  # GMT+5:30
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

st.set_page_config(page_title="GitHub Actions Dashboard", layout="wide")
st.title("🤖 GitHub Actions Dashboard – ai-autonomous/dqn-project")

# ---------------- AUTH CHECK ----------------
if not GITHUB_TOKEN:
    st.error("⚠️ Missing GitHub Token. Please set GITHUB_TOKEN as an environment variable.")
    st.stop()

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# ---------------- HELPER: Fetch logs and parse summary ----------------
def fetch_job_logs_and_summary(repo, job_id, headers):
    """
    Downloads job logs via API, saves as .txt, reads summary block, then deletes the file.
    """
    url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
    response = requests.get(url, headers=headers, allow_redirects=True)
    eval_data = {"Mean Reward": "N/A"}

    if response.status_code != 200:
        print(f"⚠️ Failed to download logs for job {job_id} (HTTP {response.status_code})")
        return eval_data

    # Save logs temporarily
    log_file_path = f"job_{job_id}_logs.txt"
    with open(log_file_path, "wb") as f:
        f.write(response.content)

    try:
        with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            match = re.search(
                r"=== Evaluation Summary ===(.*?)Mean Reward:\s*([0-9\.\+\-\s±]+)",
                text, re.S)
            if match:
                block, reward_line = match.groups()
                eval_data = dict(re.findall(r"([A-Z_]+):\s*(\d+)", block))
                eval_data["Mean Reward"] = reward_line.strip()
    except Exception as e:
        print(f"⚠️ Error reading logs for job {job_id}: {e}")
    finally:
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

    return eval_data

# ---------------- PAGINATION ----------------
if "page" not in st.session_state:
    st.session_state.page = 1

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    prev_page = st.button("⬅️ Previous", disabled=(st.session_state.page == 1))
with col3:
    next_page = st.button("Next ➡️")

if prev_page:
    st.session_state.page -= 1
if next_page:
    st.session_state.page += 1

page = st.session_state.page
url = f"https://api.github.com/repos/{REPO}/actions/runs?per_page={PER_PAGE}&page={page}"

# ---------------- FETCH WORKFLOW RUNS ----------------
response = requests.get(url, headers=HEADERS)
if response.status_code != 200:
    st.error(f"❌ GitHub API Error {response.status_code}: {response.text}")
    st.stop()

runs = response.json().get("workflow_runs", [])
if not runs:
    st.warning("⚠️ No workflow runs found on this page.")
    st.stop()

# ---------------- PARSE RUNS ----------------
rows = []
for run in runs:
    created_at = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) + TIMEZONE_OFFSET
    date_str = created_at.strftime("%b %d, %I:%M %p GMT+5:30")

    # Get jobs for this run
    jobs_url = run["jobs_url"]
    job_resp = requests.get(jobs_url, headers=HEADERS)
    if job_resp.status_code != 200:
        continue

    for job in job_resp.json().get("jobs", []):
        start = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00")) if job.get("completed_at") else start
        duration_sec = (end - start).total_seconds()
        m, s = divmod(int(duration_sec), 60)

        eval_data = fetch_job_logs_and_summary(REPO, job["id"], HEADERS)

        row = {
            "Workflow": run["name"],
            "Job": job["name"],
            "Branch": run["head_branch"],
            "Status": job["status"].capitalize(),
            "Result": job["conclusion"] or "In Progress",
            "Triggered By": run["triggering_actor"]["login"] if run.get("triggering_actor") else "N/A",
            "Created At": date_str,
            "Duration": f"{m}m {s}s",
            "View on GitHub": f"https://github.com/{REPO}/actions/runs/{run['id']}/job/{job['id']}",
        }
        row.update(eval_data)
        rows.append(row)

# ---------------- DATAFRAME ----------------
df = pd.DataFrame(rows)
if df.empty:
    st.warning("No jobs found for this page.")
    st.stop()

st.success(f"✅ Showing {len(df)} job runs from page {page} for `{REPO}`")

# ---------------- INTERACTIVE TABLE ----------------
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination(enabled=True, paginationPageSize=10)
gb.configure_default_column(editable=False, groupable=True, filter=True, sortable=True, resizable=True)
gb.configure_column(
    "View on GitHub",
    header_name="View",
    cellRenderer=(
        'function(params) {return `<a href="${params.value}" target="_blank" '
        'style="text-decoration:none;color:#1E88E5;">🔗 Open ▶️</a>`;}'
    ),
)
grid_options = gb.build()

st.write("### 📋 Job Runs")
AgGrid(
    df,
    gridOptions=grid_options,
    enable_enterprise_modules=False,
    allow_unsafe_jscode=True,
    update_mode=GridUpdateMode.NO_UPDATE,
    height=450,
    fit_columns_on_grid_load=True,
)

# ---------------- SUMMARY ----------------
st.markdown("### 📊 Summary Insights")

success_count = df[df["Result"].str.lower() == "success"].shape[0]
failed_count = df[df["Result"].str.lower() == "failure"].shape[0]
in_progress = df[df["Result"].str.lower() == "in progress"].shape[0]

st.markdown(f"""
- ✅ **Successful jobs:** {success_count}  
- ❌ **Failed jobs:** {failed_count}  
- 🕓 **In progress:** {in_progress}  
- 📄 **Page:** {page}
""")

st.caption("💡 Data fetched from GitHub API | Logs downloaded as text and parsed automatically.")
