import streamlit as st
import requests
import os
from datetime import datetime

# Load token from environment variable
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

st.set_page_config(page_title="GitHub Actions Summary", layout="centered")
st.title("🔍 GitHub Actions Summary")

repo = st.text_input("Enter your GitHub repo (format: username/repo)", "")

if not GITHUB_TOKEN:
    st.error("⚠️ Missing GitHub Token. Please set it as an environment variable.")
elif repo:
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{repo}/actions/runs"

    st.write("Fetching workflow runs...")
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error(f"❌ Error: {response.status_code} - {response.text}")
    else:
        data = response.json()
        runs = data.get("workflow_runs", [])[:5]

        if not runs:
            st.warning("No workflow runs found for this repo.")
        else:
            for run in runs:
                name = run["name"]
                status = run["status"]
                conclusion = run["conclusion"] or "In Progress"
                created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                updated = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
                duration = (updated - created).total_seconds()
                m, s = divmod(int(duration), 60)

                st.markdown(f"""
                **🧱 Workflow:** {name}  
                **🕒 Duration:** {m}m {s}s  
                **📄 Status:** `{status}`  
                **✅ Result:** `{conclusion}`  
                🔗 [View on GitHub]({run['html_url']})  
                ---  
                """)

else:
    st.info("👆 Enter a repository to see its last 5 GitHub Actions runs.")
