# Security Review & Improvement Plan – *MMRag*
_Date: 2025-05-11_

## Legend  
**[H] High risk** | **[M] Medium risk** | **[L] Low / Info**

---

## 1  Python Source Issues
### [H] server.py
* Lines 660-784 – unrestricted path ingestion via `subprocess.Popen` (user-supplied args).  
  *Risk* → Arbitrary host-path access, resource exhaustion.  
  *Mitigation* → Whitelist flags, reject absolute/parent paths, validate URLs, sandbox ingestion.

### [M] server.py
* Lines 705-724 – unbounded loop fetching channel history.  
  *Risk* → Denial-of-Service with very large channels.  
  *Mitigation* → Upper-bound pages or messages per request.

### [M] server.py
* Lines 905-909 – Flask dev server (`app.run`) used in production.  
  *Risk* → Single-threaded, missing hardening headers.  
  *Mitigation* → Run under Gunicorn/uWSGI behind reverse proxy.

### [M] server.py – SSRF via env-controlled URLs  
* If `QDRANT_URL` or `mattermost_url` are poisoned, internal metadata endpoints could be reached.  
  *Mitigation* → Allow-list valid hostnames.

### [L] server.py
* Lines 371-383 – prints secrets/model names to logs.  
  *Mitigation* → Scrub secrets before logging.

* `run_inject` leaves an undeleted `NamedTemporaryFile` on error.  
  *Mitigation* → Use `TemporaryDirectory` or unlink in `finally`.

### [L] config.py
* No hard-coded secrets (good) but relies on shared static tokens.  
  *Mitigation* → Rotate tokens or switch to HMAC-signed auth.

---

## 2  Docker / Compose
### [H] Dockerfile
* Runs as **root**; no `USER` directive.  
  *Mitigation* → `adduser app && USER app`, adjust perms.

### [M] docker-compose.yml
* Uses `qdrant/qdrant:latest` – un-pinned tag.  
  *Mitigation* → Pin to digest or semver (e.g. `qdrant/qdrant:1.9.1`).

### [M] Dockerfile + Compose
* Ports 5000 / 6333 bound to 0.0.0.0.  
  *Mitigation* → Bind to 127.0.0.1 or behind reverse proxy/VPN.

### [L] Dockerfile
* Build tools left in final image.  
  *Mitigation* → Multi-stage build to slim runtime layer.

---

## 3  Secrets & Data in Repository
### [H] `storage_llamaindex_db/**`
* Vector store & docstore JSON committed.  
  *Risk* → Potential data/PII leakage.  
  *Mitigation* → Move to external volume, add to `.gitignore`, purge history.

### [M] No direct API keys detected, but .env must stay untracked.

---

## 4  Error Handling & Defences
* Static token validation works but would benefit from rotating/HMAC tokens.  
* Timeouts are set on outbound requests (good).  
* Consider adding request rate-limiting.

---

## Overall Risk Assessment
**Internal-only deployment:** Medium risk  
**Public exposure:** High risk

### Highest-Priority Fixes
1. Non-root `USER` and multi-stage Docker image.  
2. Run Flask app under Gunicorn; bind locally.  
3. Remove `storage_llamaindex_db/**` from VCS.  
4. Harden `/inject` argument validation & sandbox.  
5. Pin all image tags/digests; tighten network exposure.

---

_Completing the above items will substantially reduce the attack surface and data-exposure risk._
