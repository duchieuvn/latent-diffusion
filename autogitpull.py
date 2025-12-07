import subprocess
import os
import time
from datetime import datetime

# ===== CONFIGURATION =====
repo_path = "./"  # local path to your Git repository
branch = "main"                   # branch to track
check_interval = 10               # seconds between checks
service_name = None       # optional: systemd service to restart, set None if not needed
log_file = "./git_pull.log"  # log file path

# ===== HELPER FUNCTION =====
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    # print(full_message)
    with open(log_file, "a") as f:
        f.write(full_message + "\n")

# ===== MAIN LOOP =====
os.chdir(repo_path)
log("Git pull watcher started.")
print("Git pull watcher started.")

while True:
    try:
        # Fetch latest commits from remote
        subprocess.run(["git", "fetch", "origin"], check=True)

        # Get local and remote commit hashes
        local_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
        remote_commit = subprocess.check_output(
            ["git", "rev-parse", f"origin/{branch}"]
        ).decode("utf-8").strip()

        if local_commit != remote_commit:
            log(f"New commits detected ({remote_commit}). Pulling changes...")
            subprocess.run(["git", "pull", "origin", branch], check=True)
            
            if service_name:
                log(f"Restarting service '{service_name}'...")
                subprocess.run(["systemctl", "restart", service_name], check=True)
            
            log("Update completed.")
        # else:
        #     log("No new commits found.")

    except subprocess.CalledProcessError as e:
        log(f"Error during git operation or service restart: {e}")

    # Wait before next check
    time.sleep(check_interval)
