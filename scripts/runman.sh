#!/usr/bin/env bash
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$PROJECT_ROOT/chatbot-ui"
ENV_FILE="$APP_DIR/.env.local"

# ── Switches ─────────────────────────────────────────────────────────────────────
UI_MODE="${UI_MODE:-flask}"            # flask | gradio
START_FRONTEND="${START_FRONTEND:-1}"  # 1 to run Next.js dev server too

# ── Helpers ─────────────────────────────────────────────────────────────────────
put_kv() {
  # put_kv <file> <KEY> <VALUE>
  local file="$1" key="$2" val="$3"
  python3 - "$file" "$key" "$val" <<'PY'
import sys
from pathlib import Path
f,k,v = sys.argv[1:]
p = Path(f)
lines = p.read_text().splitlines() if p.exists() else []
for i, line in enumerate(lines):
    if line.startswith(f"{k}="):
        lines[i] = f"{k}={v}"
        break
else:
    lines.append(f"{k}={v}")
p.write_text("\n".join(lines) + "\n")
PY
}

ensure_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "$APP_DIR/.env.local.example" ]]; then
      cp "$APP_DIR/.env.local.example" "$ENV_FILE"
    else
      : > "$ENV_FILE"
    fi
  fi
}

strip_ansi() {
  # read stdin, strip ANSI, print to stdout
  perl -pe 's/\x1B\[[0-9;]*[A-Za-z]//g'
}

# ── Supabase: status → parse; else start ─────────────────────────────────────────
run_supabase_and_capture() {
  cd "$APP_DIR"
  local out rc
  set +e
  out="$(npx supabase status 2>&1)"
  rc=$?
  set -e

  if (( rc != 0 )) || [[ "$out" != *"API URL:"* ]]; then
    echo "[manual] starting Supabase via npx..."
    out="$(npx supabase start 2>&1)"
  else
    echo "[manual] Supabase already running; reusing."
  fi

  # save raw & return cleaned
  printf '%s\n' "$out" > "$PROJECT_ROOT/.supabase_status_raw.log"
  printf '%s\n' "$out" | strip_ansi
}

# ── Parse with awk/sed (no JSON, no subshell function piping issues) ────────────
parse_supabase_clean_text() {
  # stdin: cleaned CLI text
  # stdout: three lines: URL, ANON, SERVICE (empty if not found)
  awk '
    BEGIN{ url=""; anon=""; svc="" }
    /^[[:space:]]*API URL:[[:space:]]*/       { sub(/^[[:space:]]*API URL:[[:space:]]*/,"");       if(url=="")  url=$0 }
    /^[[:space:]]*Publishable key:[[:space:]]*/{ sub(/^[[:space:]]*Publishable key:[[:space:]]*/,""); if(anon=="") anon=$0 }
    /^[[:space:]]*Secret key:[[:space:]]*/     { sub(/^[[:space:]]*Secret key:[[:space:]]*/,"");     if(svc=="")  svc=$0 }
    END{
      print url
      print anon
      print svc
    }
  '
}

apply_env_to_next_from_text() {
  local clean_text="$1"

  # primary parse
  local API_URL ANON SERVICE
  API_URL="$(printf '%s\n' "$clean_text" | parse_supabase_clean_text | sed -n '1p')"
  ANON="$(    printf '%s\n' "$clean_text" | parse_supabase_clean_text | sed -n '2p')"
  SERVICE="$( printf '%s\n' "$clean_text" | parse_supabase_clean_text | sed -n '3p')"

  # fallbacks (very forgiving)
  if [[ -z "$API_URL" ]]; then
    API_URL="$(printf '%s\n' "$clean_text" | grep -Eo 'http://127\.0\.0\.1:[0-9]{2,5}(/[A-Za-z0-9._~:/?#\[\]@!$&'\''()*+,;=%-]*)?' | head -n1 || true)"
  fi
  if [[ -z "$ANON" ]]; then
    ANON="$(printf '%s\n' "$clean_text" | grep -Eo '\bsb_publishable_[A-Za-z0-9_\-]+' | head -n1 || true)"
  fi
  if [[ -z "$SERVICE" ]]; then
    SERVICE="$(printf '%s\n' "$clean_text" | grep -Eo '\bsb_secret_[A-Za-z0-9_\-]+' | head -n1 || true)"
  fi

  if [[ -n "${SUPABASE_EXTERNAL_HOST:-}" ]]; then
    API_URL="$(python3 - "$API_URL" "$SUPABASE_EXTERNAL_HOST" <<'PY'
import sys
from urllib.parse import urlparse, urlunparse

api_url, new_host = sys.argv[1:]
parsed = urlparse(api_url)
host = parsed.netloc or ""
if host:
    if ":" in host:
        port = host.split(":", 1)[1]
        netloc = f"{new_host}:{port}"
    else:
        netloc = new_host
    updated = parsed._replace(netloc=netloc)
    print(urlunparse(updated))
else:
    print(f"http://{new_host}")
PY
)"
  fi

  if [[ -z "$API_URL" || -z "$ANON" || -z "$SERVICE" ]]; then
    echo "[manual] Failed to parse Supabase output."
    echo "  Raw log: $PROJECT_ROOT/.supabase_status_raw.log"
    echo "  Clean log preview:"
    echo "------------------------------------------------------------"
    printf '%s\n' "$clean_text" | head -n 120
    echo "------------------------------------------------------------"
    exit 1
  fi

  ensure_env_file
  put_kv "$ENV_FILE" "NEXT_PUBLIC_SUPABASE_URL" "$API_URL"
  put_kv "$ENV_FILE" "NEXT_PUBLIC_SUPABASE_ANON_KEY" "$ANON"
  put_kv "$ENV_FILE" "SUPABASE_SERVICE_ROLE_KEY" "$SERVICE"

  # optional passthroughs if exported
  if [[ -n "${OPENAI_API_BASE_URL:-}" ]]; then
    put_kv "$ENV_FILE" "OPENAI_API_BASE_URL" "$OPENAI_API_BASE_URL"
  fi
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    put_kv "$ENV_FILE" "OPENAI_API_KEY" "$OPENAI_API_KEY"
  fi

  echo "[manual] parsed:"
  echo "  url=$API_URL"
  echo "  anon=${ANON:0:28}…"
  echo "  service=${SERVICE:0:28}…"
}

# ── Backend / Frontend ──────────────────────────────────────────────────────────
start_backend() {
  cd "$PROJECT_ROOT"
  if [[ "$UI_MODE" == "gradio" ]]; then
    export GRADIO_HOST="${GRADIO_HOST:-0.0.0.0}"
    export GRADIO_PORT="${GRADIO_PORT:-7860}"
    export GRADIO_CONCURRENCY="${GRADIO_CONCURRENCY:-${CONCURRENT_REQUESTS:-4}}"
    python3 gradio_app.py
  else
    export BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
    export BACKEND_PORT="${BACKEND_PORT:-5000}"
    python3 agentwithUi.py
  fi
}

start_frontend() {
  cd "$APP_DIR"
  export NEXT_TELEMETRY_DISABLED=1
  local host="${CHATBOT_HOST:-0.0.0.0}"
  local port="${CHATBOT_PORT:-3000}"
  if [[ ! -d node_modules ]]; then npm install; fi
  npm run dev -- --hostname "$host" --port "$port"
}

# ── Main ────────────────────────────────────────────────────────────────────────
CLEAN_TEXT="$(run_supabase_and_capture)"

apply_env_to_next_from_text "$CLEAN_TEXT"

# Export .env.local into current shell (optional; backend may read env directly)
set -a
# shellcheck disable=SC1091
. "$ENV_FILE"
set +a

echo "[manual] Supabase ready:"
echo "  API:    $NEXT_PUBLIC_SUPABASE_URL"
if [[ -n "${SUPABASE_EXTERNAL_HOST:-}" ]]; then
  echo "  Studio: http://${SUPABASE_EXTERNAL_HOST}:54323"
  echo "  DB:     postgresql://postgres:postgres@${SUPABASE_EXTERNAL_HOST}:54322/postgres"
else
  echo "  Studio: http://127.0.0.1:54323"
  echo "  DB:     postgresql://postgres:postgres@127.0.0.1:54322/postgres"
fi

echo "[manual] starting backend..."
start_backend & BACKEND_PID=$!

if [[ "${START_FRONTEND}" == "1" ]]; then
  echo "[manual] starting frontend..."
  start_frontend & FRONTEND_PID=$!
fi

echo "Backend PID: ${BACKEND_PID:-}"
echo "Frontend PID: ${FRONTEND_PID:-}"

cleanup() {
  [[ -n "${FRONTEND_PID:-}" ]] && kill "${FRONTEND_PID}" 2>/dev/null || true
  [[ -n "${BACKEND_PID:-}" ]] && kill "${BACKEND_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait
