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
* **React 19 Conflicts:** The UI uses Next.js 15 & React 19. Many older packages will throw Peer Dependency errors during a build.ially - actually their docs says 
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

## ⚠️ 7. Coolify Proxy Route Loss (The "Gateway Timeout After Every Deploy" Bug)

**This is a known Coolify bug.** Every time you redeploy, Coolify's Traefik proxy has a race condition where it may lose its routing config for your containers. External traffic gets a `504 Gateway Timeout`. The UI still appears to work because the Next.js UI server calls the API via the **internal Docker network** (`http://api:8000`), bypassing Traefik entirely — so you won't notice the bug from the UI.

### How to detect it:
```bash
curl -sf "https://base-api.muktam.online/api/v1/health"
# If this returns 504, the proxy has lost the route. If 200, you're fine.
```

Also visible in logs: API logs only show `127.0.0.1` (docker healthchecks), zero requests from `10.0.3.x` (external via proxy).

### Immediate fix:
```bash
docker restart coolify-proxy
```

### Permanent fix — Auto-restart proxy after route loss (add to server crontab):
```bash
crontab -e
# Add this line — checks every 5 minutes, restarts proxy if health fails:
*/5 * * * * curl -sf --max-time 10 "https://base-api.muktam.online/api/v1/health" > /dev/null || (docker restart coolify-proxy && echo "$(date): Proxy restarted due to 504" >> /var/log/proxy-watchdog.log)
```

After adding the cron job, confirm it's active:
```bash
crontab -l
```

### Why does this happen?
When Coolify recreates a container during deploy, it sends Docker events to Traefik to update routing. Due to timing issues (container starts up before Traefik processes the event, or network reconnection hiccups), Traefik loses the backend and starts returning 504. Restarting the proxy forces it to re-scan all running containers and rebuild its routing table.

### After every redeploy — manual checklist:
1. Wait ~60s for containers to be healthy
2. Run `curl -sf "https://base-api.muktam.online/api/v1/health"` 
3. If 504 → `docker restart coolify-proxy`
4. Wait 10s → test again

## 🔌 8. New Provider Integration Checklist

Every time we integrate a new external service — LLM, TTS, STT, storage, webhook, anything — the same class of bugs bites us. This checklist is distilled from real debugging sessions. Use it **before** writing code, not after.

### Step 0: Get a Working API Call First
Before writing a single line of integration code, make a **standalone working call** to the provider's API (curl, Postman, or a throwaway Python script). Save the exact request and response. This is your ground truth.

* Paste their official example payload as a comment at the top of your service file.
* Note every field's **exact type**. APIs are picky — `1` (integer) and `true` (boolean) serialize differently in JSON, and a provider may silently reject the wrong type with zero error messages.
* Confirm the **exact field names** the API returns (e.g. `voiceId` vs `voice_id`, `model_name` vs `modelName`). Never assume — always log the raw response.

### Step 1: Trace the Full Data Flow
Before debugging anything at the service layer, trace every user-facing value from the **UI** all the way to the **outgoing API payload**. Bugs hide at every hop:

* **UI → Backend:** Is the UI sending the API's internal identifier (e.g. a lowercase slug) or the human display label? Many APIs are **case-sensitive** and will silently fail on wrong casing.
* **Backend → Config DB:** Does the Pydantic model coerce the value? A `bool` field stores as JSON `false`, but the API might need integer `0`. A `str` field might store a full language name when the API needs an ISO code.
* **Config DB → Service Constructor:** Do `getattr()` fallbacks use the correct defaults, or stale ones from a previous iteration?
* **Service → API Payload:** Log the **full serialized JSON** right before sending. Compare it **field-by-field** against the working curl from Step 0. This single step would have caught every bug in our history.

### Step 2: Understand Streaming & Protocol Semantics
WebSocket, SSE, and streaming APIs have **control flags** that fundamentally change behavior:

* Some APIs have "continue", "stream", or "partial" flags — setting them wrong means the API **buffers your input and waits forever** for more data, returning nothing. This creates the maddening symptom: "connection succeeds, payload sends, zero response."
* Some APIs require explicit "flush", "end", or "close" signals, or they'll hold data in a buffer indefinitely.
* **Rule of thumb:** If you're sending a complete, self-contained request (not a partial stream), make sure all flags tell the API "this is complete, process it now." Don't blindly set streaming flags to `true` without understanding what happens downstream.

### Step 3: Match the Framework's Internal Signatures
Our upstream framework (Pipecat) evolves between versions. Before writing any service class:

* Check where the **base class** currently lives — modules get renamed between versions. Use try/except imports to support both old and new paths gracefully.
* Check the **method signatures** of the base class you're extending. If the upstream added or changed a parameter, your override must match.
* Symptom of a signature mismatch: `takes X positional arguments but Y were given`. Always inspect the parent class signature before writing your own.

### Step 4: Type Discipline in Payloads
Provider APIs are strict about JSON types even when their docs don't say so:

| Python type | JSON output | Gotcha |
|------------|-------------|--------|
| `True` / `False` | `true` / `false` | Some APIs want `1` / `0` (integer) |
| `1.0` (float) | `1.0` | Some APIs want `1` (no decimal) for certain fields |
| `"Hindi"` | `"Hindi"` | API probably wants `"hi"` (ISO code) |

* **Cast explicitly** in the payload builder: `int(...)`, `float(...)`, `.lower()`.
* **Store API-native values** in config — language codes not display names, ID slugs not human labels.

### Step 5: Silent Failures Are the Norm
The most dangerous provider bugs produce **no error at all**:

* Connection succeeds ✅, payload sends ✅, but zero response. This means your payload is **semantically** wrong (wrong types, wrong control flags, wrong identifier format), not syntactically broken. The provider accepted your JSON structure but quietly ignored it.
* **Always log on every received message** (type, size, status). If you see zero received-message logs after a successful send, stop debugging your connection code — your payload semantics are wrong. Go back to Step 0 and compare field-by-field.

### Step 6: Default Config Values Must Match Provider Docs
When adding a new provider to `registry.py`:

* Copy default values **exactly** from the provider's API docs example. Don't guess "reasonable" values — what seems reasonable may cause silent failure.
* When you later fix defaults, remember that **existing user configs in the DB still have the old values**. Add migration logic in the Pydantic `model_validator` to convert legacy values on load (see `UserConfiguration.strip_deprecated_providers` for the pattern).

### Step 7: API Response → UI Mapping
When building dropdowns or selection lists from a provider's API:

* **Never assume field names.** Always log the raw response first: `logger.debug(f"Raw API response: {data}")`.
* Common mapping traps: camelCase vs snake_case, nested objects vs flat fields, arrays vs single strings.
* What you store in config must be the **API's internal identifier**, not the display name the UI shows. The UI renders a human label; the config stores a machine slug.

---
**SUMMARY:**
When updating the app: Fetch from upstream -> `git rebase` -> Fix conflicts with isolated plugins -> Enforce RAM ceilings -> Check Database Migrations -> Force Deploy.

When integrating a new provider: Get a working API call first → Log raw responses → Trace the full data flow → Match framework signatures → Cast types explicitly → Compare your full payload against docs → Check streaming/control flags.
