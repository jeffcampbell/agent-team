# AI-Assisted Setup

This file is written for an AI coding agent (Claude Code, Cursor, etc.) to follow. It will configure Yamanote to manage your project.

**Human:** just tell your LLM to "follow SETUP.md" and it should do the rest.

---

## Step 1: Check prerequisites

Run these commands and verify they succeed:

```bash
python3 --version   # need 3.11+
git --version
claude --version     # Claude Code CLI must be installed and authenticated
```

If `claude` is not found, stop and tell the user to install Claude Code first: https://docs.anthropic.com/en/docs/claude-code

## Step 2: Detect the target project

Look for candidate projects to manage. Search for git repositories under `~/Development` (or the user's typical code directory):

```bash
find ~/Development -maxdepth 2 -name ".git" -type d 2>/dev/null | sed 's|/.git$||'
```

Present the list to the user and ask:
1. **Which project should agent-team manage?** (pick one from the list, or let them type a path)
2. **Should agent-team restart a service after merging?** If yes, ask for the systemd service name (e.g. `my-app.service`). If no, leave it blank.
3. **Would you like to enable the web dashboard?** It provides a real-time status page showing agent activity, pipeline progress, and backlog — accessible from any device on the local network. If yes, ask what port to use (default: `8080`). If no, skip it.

## Step 3: Write the `.env` file

Copy `.env.example` to `.env` and fill in the values from Step 2:

```bash
cp .env.example .env
```

Edit `.env` with the detected/confirmed values. For example, if the user chose `~/Development/my-app` with service `my-app.service`:

```
AGENT_TEAM_DEV_DIR=~/Development
AGENT_TEAM_DEFAULT_PROJECT=my-app
AGENT_TEAM_SERVICE_RESTART_CMD=sudo systemctl restart my-app.service
```

- `AGENT_TEAM_DEV_DIR` = the parent directory (e.g. `~/Development`)
- `AGENT_TEAM_DEFAULT_PROJECT` = just the directory name, not the full path (e.g. `my-app`)
- `AGENT_TEAM_SERVICE_RESTART_CMD` = leave empty or omit entirely if no service restart is needed
- `AGENT_TEAM_DASHBOARD_PORT` = the port number if the user wants the dashboard (e.g. `8080`), or omit/set to `0` to disable

## Step 4: Validate

```bash
source .env && python3 -c "import config; import orchestrator; print('OK')"
```

Also confirm the resolved project directory exists:

```bash
source .env && python3 -c "
import config, os
project = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)
assert os.path.isdir(project), f'Directory not found: {project}'
assert os.path.isdir(os.path.join(project, '.git')), f'Not a git repo: {project}'
print(f'Target project: {project}')
print(f'Service restart: {config.SERVICE_RESTART_CMD or \"(disabled)\"}')
"
```

If either check fails, go back to Step 3 and fix the values.

## Step 5: Set up systemd service (optional)

Ask the user if they want to run Yamanote as a systemd service. If yes:

1. Read `agent-team.service` and update the paths to match this machine:
   - `User=` should be the current user (`whoami`)
   - `WorkingDirectory=` should be the absolute path to this agent-team repo
   - `ExecStart=` should point to `orchestrator.py` in this repo
   - `EnvironmentFile=` should point to the `.env` file in this repo
   - `Environment=PATH=` must include the directory containing `claude` (run `which claude` to find it)

2. Copy and enable:
   ```bash
   sudo cp agent-team.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable agent-team
   ```

3. If `AGENT_TEAM_SERVICE_RESTART_CMD` uses `sudo`, set up passwordless sudo:
   ```bash
   # Replace <user> and <service> with actual values
   echo '<user> ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart <service>' | sudo tee /etc/sudoers.d/agent-team
   sudo chmod 440 /etc/sudoers.d/agent-team
   ```

4. Start it:
   ```bash
   sudo systemctl start agent-team
   systemctl status agent-team
   ```

## Step 6: Confirm it's working

If running as a service:
```bash
journalctl -u agent-team -n 20 --no-pager
```

If running manually:
```bash
source .env && python3 orchestrator.py
# Watch for the ORCHESTRATOR STARTING banner, then Ctrl+C to stop
```

Tell the user: "Yamanote is set up and managing `<project-name>`. It will start generating specs and implementing features autonomously. Monitor progress with `tail -f agents/activity.log`."
