# MUKTAM Platform - Core Developer Guide & Constraints

**🚨 CRITICAL WARNING TO ANY FUTURE DEVELOPERS OR AI AGENTS 🚨**
Before you install new dependencies, merge from `dograh/upstream`, or deploy this codebase, **YOU MUST READ AND FOLLOW THESE RULES.**

This repository is a heavily diverged "Hard Fork" of `dograh-hq/dograh`. We have custom performance constraints, a custom Sarvam LLM pipeline, and unique YAML deployment rules.

---

## 🏗️ 1. The Modularity Rule (Plugin Architecture Only!)
To survive future rebases, **NEVER** scatter custom logic across the upstream core files.
* **Bad:** Adding 500 lines of Sarvam code inside `api/services/pipecat/service_factory.py`.
* **Good:** Putting the code in `api/services/sarvam/llm.py` and adding a *single* `import` line to `service_factory.py`.
* Always write custom features as isolated, standalone modules. Only touch upstream files when strictly necessary to hook your plugin in.

## 🧠 2. UI Memory Restraints (The 2GB Rule)
Our deployment VPS lacks the memory of a large corporate server. The UI builder container **will OOM crash the server** if left unrestricted.
* **Dockerfile Location:** `ui/Dockerfile`
* **The Rule:** You MUST keep `ENV NODE_OPTIONS="--max-old-space-size=2048"` intact in the UI Builder stage. Do not increase it to `4096`. Do not delete it.

## 📦 3. strict dependency requirements
* **React 19 Conflicts:** The UI uses Next.js 15 & React 19. Many older packages will throw Peer Dependency errors during a build.
    * **The Fix:** We exclusively use `npm ci --legacy-peer-deps` or `npm install --legacy-peer-deps` in the `ui/Dockerfile`. Do NOT revert to standard installs.
* **Pipecat Patches:** Python requires `wait_for2==0.4.1` for Pipecat execution. If you wipe `api/requirements.txt`, ensure `wait_for2` is placed back in.

## 🗄️ 4. Alembic Migrations ("Multiple Heads" Conflict)
If you pull from Upstream, and they added a new database table, you will see a `Multiple head revisions` error when deploying.
* **Why:** The migration chain diverged. (Upstream added a link, and we previously added a link on the same older node).
* **The Fix:**
  1. Run `cd api && PYTHONPATH=. alembic heads`
  2. Locate our custom table migration file in `api/alembic/versions/`.
  3. Modify `down_revision` in our file to literally equal the newest Revision ID that Upstream added.
  4. Now they form a clean line again!

## 🐳 5. YAML Overwrites
We significantly cleaned up `docker-compose.yaml` (e.g., removing `cloudflared`). If you rebase or merge from upstream, Git will try to re-insert their `cloudflared` services. Always manually delete those services from the YAML during a conflict resolution.

---
**SUMMARY:**
When updating the app: Fetch from upstream -> `git rebase` -> Fix conflicts with isolated plugins -> Enforce RAM ceilings -> Check Database Migrations -> Force Deploy.
