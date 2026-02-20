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
* **Node Options:** You MUST keep `ENV NODE_OPTIONS="--max-old-space-size=2048"` intact in the UI Builder stage. Do not increase it to `4096`. Do not delete it.
* **Sentry Source Maps:** Next.js uses absurd amounts of RAM generating source maps during `npm run build` (triggering Linux exit code 134 OOM Killer). In `ui/next.config.ts`, `sourcemaps: { disable: true }` and `widenClientFileUpload: false` must remain active.
* **Linting During Build:** We bypass Next.js default linting and type-checking during the Docker build stage to save RAM. In `ui/next.config.ts`, `eslint: { ignoreDuringBuilds: true }` and `typescript: { ignoreBuildErrors: true }` must be present.

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
* **Robust Schema Recovery:** `dograh` utilizes an `api/fix_db.py` script on startup to forcefully overwrite missing Alembic head revisions.
  * Because Alembic blindly trusts `api/fix_db.py` forcing the migration number, any schema differences will be permanently skipped unless dynamically checked. 
  * If you edit database models, `api/fix_db.py` MUST query `information_schema.columns` via PostgreSQL async engine to accurately verify those columns manually exist before forcibly leaping ahead in migration history. 
  * Upgrade migration blocks inside `api/alembic/versions/` must use `sa.inspect(op.get_bind())` to check if columns physically exist before calling `op.add_column`. Using `try...except` to catch `DuplicateColumn` will completely abort PostgreSQL's internal database transactions and permanently crash the deployment.

## 🐳 5. YAML Overwrites
We significantly cleaned up `docker-compose.yaml` (e.g., removing `cloudflared`). If you rebase or merge from upstream, Git will try to re-insert their `cloudflared` services. Always manually delete those services from the YAML during a conflict resolution.
* **CRITICAL Build Blocks:** Upstream removed the `build:` blocks from `api` and `ui` in `docker-compose.yaml` to rely on remote registries. **You MUST ensure the `build:` property exists** in both the `api` and `ui` blocks, otherwise Coolify will simply download their vanilla factory images instead of compiling your custom constraints and `api/fix_db.py` fixes!

## 🌐 6. Next.js Docker Networking Constraints
Coolify and Alpine Linux handle networking differently than a raw Ubuntu server. Follow these rules or the website UI will crash and Coolify will drop the traffic.
* **The Alpine IPv6 Trap:** In Alpine Linux, `localhost` resolves to an IPv6 `::1` address. Next.js natively binds to IPv4. Therefore, Docker healthchecks using `wget http://localhost:3010` **will fail**. Your Docker Compose health check must explicitly use `http://127.0.0.1:3010` or else Coolify will see the UI container as 'unhealthy' and sever public access.
* **UI Server Binding:** In `ui/Dockerfile`, when running Next.js standalone, the command MUST specify `HOSTNAME=0.0.0.0` (e.g., `CMD sh -c "HOSTNAME=0.0.0.0 PORT=3010 node server.js"`). If omitted, Next.js blocks incoming public traffic.
* **API Client Backend Discovery:** `dograh/upstream`'s `route.ts` API client blindly falls back to pulling from internal Docker networks (e.g. `http://api:8000`) and passes that string back to the user's web browser as `http://localhost:8000`, causing `CONNECTION_REFUSED` on login. In `ui/src/app/api/config/version/route.ts`, if the environment is a docker internal IP, the `clientApiBaseUrl` must be `null` so the browser gracefully falls back to `window.location.origin` without guessing port numbers.
---
**SUMMARY:**
When updating the app: Fetch from upstream -> `git rebase` -> Fix conflicts with isolated plugins -> Enforce RAM ceilings -> Check Database Migrations -> Force Deploy.
