#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)

REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
REDIS_INSTALL_DIR="${REDIS_INSTALL_DIR:-$HOME/.local/redis}"
SUPABASE_CLI_BIN_DIR="${SUPABASE_CLI_BIN_DIR:-$HOME/.supabase/bin}"  # kept for compatibility
PATH="$REDIS_INSTALL_DIR/bin:$SUPABASE_CLI_BIN_DIR:$PATH"

REDIS_HOST=$(echo "$REDIS_URL" | sed -E 's|^[^/]+://([^:/]+).*|\1|')
REDIS_PORT=$(echo "$REDIS_URL" | sed -E 's|^[^/]+://[^:/]+:([0-9]+).*|\1|')
if [[ "$REDIS_PORT" == "$REDIS_URL" ]]; then
  REDIS_PORT="6379"
fi

UI_MODE="${UI_MODE:-flask}"
if [[ "$UI_MODE" == "gradio" ]]; then
  START_FRONTEND_DEFAULT="0"
else
  START_FRONTEND_DEFAULT="1"
fi
START_FRONTEND="${START_FRONTEND:-$START_FRONTEND_DEFAULT}"
case "${START_FRONTEND}" in
  1|true|TRUE|yes|YES|on|ON) START_FRONTEND="1" ;;
  0|false|FALSE|no|NO|off|OFF) START_FRONTEND="0" ;;
  *) START_FRONTEND="1" ;;
esac

# Supabase startup mode
SUPABASE_MODE="${SUPABASE_MODE:-}"
if [[ -z "$SUPABASE_MODE" ]]; then
  if [[ -n "${USE_LOCAL_SUPABASE:-}" ]]; then
    case "${USE_LOCAL_SUPABASE}" in
      1|true|TRUE|yes|YES|on|ON) SUPABASE_MODE="cli" ;;
      0|false|FALSE|no|NO|off|OFF) SUPABASE_MODE="manual" ;;
      *) SUPABASE_MODE="cli" ;;
    esac
  else
    SUPABASE_MODE="cli"
  fi
fi
case "$SUPABASE_MODE" in
  cli|CLI) SUPABASE_MODE="cli" ;;
  local|LOCAL) SUPABASE_MODE="local" ;;
  manual|MANUAL|remote|REMOTE) SUPABASE_MODE="manual" ;;
  *)
    echo "[start_env] Unrecognized SUPABASE_MODE='$SUPABASE_MODE'. Falling back to 'manual'." >&2
    SUPABASE_MODE="manual"
    ;;
esac

if [[ "$UI_MODE" != "gradio" ]]; then
  export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost:3000}"
fi

check_redis() {
  if command -v redis-cli >/dev/null 2>&1; then
    redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1 && return 0
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1 && return 0
  fi
  return 1
}

# ---------------- Docker helpers (prefer existing; fallback to Lima) ----------------
prefer_lima_context() {
  # Context switching doesn't work if DOCKER_HOST is set
  unset DOCKER_HOST
  # If a 'lima' context exists, prefer it; otherwise keep current
  if docker context ls --format '{{.Name}}' 2>/dev/null | grep -q '^lima$'; then
    docker context use lima >/dev/null 2>&1 || true
  fi
}

docker_ready() {
  docker info >/dev/null 2>&1
}

create_or_use_lima_context() {
  # If Lima's socket exists, ensure a context points to it and is active
  local sock
  if [[ -n "${LIMA_HOME:-}" && -S "$LIMA_HOME/docker/sock/docker.sock" ]]; then
    sock="$LIMA_HOME/docker/sock/docker.sock"
  else
    # Try to discover the socket via limactl
    if command -v limactl >/dev/null 2>&1; then
      sock="$(limactl list docker --format '{{.Dir}}/sock/docker.sock' 2>/dev/null || true)"
    fi
  fi
  if [[ -n "${sock:-}" && -S "$sock" ]]; then
    docker context rm -f lima >/dev/null 2>&1 || true
    docker context create lima --docker "host=unix://$sock" >/dev/null 2>&1 || true
    docker context use lima >/dev/null 2>&1 || true
  fi
}

ensure_lima_vm() {
  # Ensure we have a Lima VM named "docker" running, using template if needed
  if ! command -v limactl >/dev/null 2>&1; then
    echo "[start_env] Docker daemon not reachable and 'limactl' not found. Start Docker Desktop/Colima or install Lima." >&2
    return 1
  fi

  # Prefer APFS path if user provided LIMA_HOME; don't force-change it here.
  if limactl list 2>/dev/null | awk 'NR>1 && $1=="docker"{f=1} END{exit !f}'; then
    limactl start docker >/dev/null 2>&1 || true
  else
    # Try built-in template first (newer Lima)
    if ! limactl start --name=docker template://docker >/dev/null 2>&1; then
      # Fall back to fetching a current template if template://docker is unavailable
      tmp_yaml="/tmp/lima-docker.$$.yaml"
      # Use 'templates' path (new Lima repo layout)
      curl -fsSL -o "$tmp_yaml" "https://raw.githubusercontent.com/lima-vm/lima/master/templates/docker.yaml" || {
        echo "[start_env] Failed to fetch Lima docker template." >&2
        return 1
      }
      limactl start --name=docker "$tmp_yaml" >/dev/null 2>&1 || true
      rm -f "$tmp_yaml"
    fi
  fi
  # After starting, (re)create lima context if we can find the socket
  create_or_use_lima_context
}

ensure_docker_engine() {
  prefer_lima_context
  if docker_ready; then return 0; fi

  # Try to start or create Lima VM and context
  ensure_lima_vm || true
  prefer_lima_context
  if docker_ready; then return 0; fi

  echo "[start_env] Docker daemon is not reachable. Start Docker (Desktop, Lima, or Colima) and retry." >&2
  exit 1
}
# ------------------------------------------------------------------------------------

set_env_var_in_file() {
  local file="$1"
  local key="$2"
  local value="$3"
  python3 - <<'PY' "$file" "$key" "$value"
import sys
from pathlib import Path

file_path, key, value = sys.argv[1:]
path = Path(file_path)
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
else:
    lines = path.read_text().splitlines()

for idx, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[idx] = f"{key}={value}"
        break
else:
    lines.append(f"{key}={value}")

path.write_text("\n".join(lines) + "\n")
PY
}

install_redis_mac_arm() {
  if command -v redis-server >/dev/null 2>&1; then
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    echo "Installing Redis via Homebrew..."
    brew update >/dev/null 2>&1 || true
    brew install redis
    return $?
  fi
  echo "Homebrew not found. Attempting local build of Redis..."
  tmpdir=$(mktemp -d)
  curl -L -o "$tmpdir/redis.tar.gz" https://download.redis.io/redis-stable.tar.gz
  tar -xzf "$tmpdir/redis.tar.gz" -C "$tmpdir"
  orig_dir=$(pwd)
  cd "$tmpdir/redis-stable"
  make >/dev/null
  make PREFIX="$REDIS_INSTALL_DIR" install >/dev/null
  cd "$orig_dir"
  rm -rf "$tmpdir"
  mkdir -p "$REDIS_INSTALL_DIR/bin"
  PATH="$REDIS_INSTALL_DIR/bin:$PATH"
  export PATH
  return 0
}

start_redis() {
  if ! command -v redis-server >/dev/null 2>&1; then
    if [[ "$OS_NAME" == "Darwin" && "$ARCH_NAME" == "arm64" ]]; then
      echo "redis-server not found. Attempting to install for macOS ARM..."
      if ! install_redis_mac_arm; then
        echo "Automatic Redis installation failed. Please install Redis manually or set REDIS_URL." >&2
        exit 1
      fi
    else
      echo "redis-server not found on PATH. Please install Redis or set REDIS_URL to a reachable instance." >&2
      exit 1
    fi
  fi
  echo "Starting Redis server..."
  redis-server --daemonize yes
  sleep 1
}

install_python_requirements() {
  if [[ ! -f requirements.txt ]]; then
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    if python3 -m pip --version >/dev/null 2>&1; then
      echo "Installing Python dependencies from requirements.txt..."
      python3 -m pip install --upgrade pip >/dev/null 2>&1 || true
      python3 -m pip install -r requirements.txt
    else
      echo "pip is not available for python3. Skipping requirements install." >&2
    fi
  else
    echo "python3 not found on PATH. Skipping requirements install." >&2
  fi
}

install_frontend_dependencies() {
  if [[ "$START_FRONTEND" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "$PROJECT_ROOT/chatbot-ui" ]]; then
    echo "chatbot-ui directory not found. Set START_FRONTEND=0 to skip starting the frontend." >&2
    exit 1
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm not found on PATH. Unable to install frontend dependencies." >&2
    exit 1
  fi
  if [[ ! -d "$PROJECT_ROOT/chatbot-ui/node_modules" || "${FORCE_FRONTEND_INSTALL:-0}" == "1" ]]; then
    echo "Installing frontend dependencies (chatbot-ui)..."
    (cd "$PROJECT_ROOT/chatbot-ui" && npm install)
  fi
}

# ---------- Supabase (no Homebrew, no global npm) ----------
# Order of attempts:
# 1) Local devDependency via npx (project dir)
# 2) Ephemeral npx (optionally pinned: SUPABASE_VERSION="2.47.2")
# 3) Go install fallback (no brew)
pick_supabase_cmd() {
  if [[ -d "$PROJECT_ROOT/chatbot-ui" ]]; then
    SUPABASE_PROJECT_DIR="$PROJECT_ROOT/chatbot-ui"
  else
    SUPABASE_PROJECT_DIR="$PROJECT_ROOT"
  fi

  if [[ -n "${SUPABASE_CMD_OVERRIDE:-}" ]]; then
    SUPABASE_CMD="$SUPABASE_CMD_OVERRIDE"
    export SUPABASE_CMD SUPABASE_PROJECT_DIR
    return 0
  fi

  have_node_npm=0
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    have_node_npm=1
  fi

  if [[ $have_node_npm -eq 1 ]]; then
    if (cd "$SUPABASE_PROJECT_DIR" && npm ls supabase --depth=0 >/dev/null 2>&1); then
      SUPABASE_CMD="(cd \"$SUPABASE_PROJECT_DIR\" && npx --yes supabase)"
      if eval "$SUPABASE_CMD --version" >/dev/null 2>&1; then
        export SUPABASE_CMD SUPABASE_PROJECT_DIR
        return 0
      fi
    fi
  fi

  if [[ $have_node_npm -eq 1 ]]; then
    if [[ -n "${SUPABASE_VERSION:-}" ]]; then
      SUPABASE_CMD="npx --yes -p supabase@${SUPABASE_VERSION} supabase"
    else
      SUPABASE_CMD="npx --yes -p supabase supabase"
    fi
    if eval "$SUPABASE_CMD --version" >/dev/null 2>&1; then
      export SUPABASE_CMD SUPABASE_PROJECT_DIR
      return 0
    fi
  fi

  if command -v go >/dev/null 2>&1; then
    echo "Installing Supabase CLI via Go modules (no brew)..."
    if go install github.com/supabase/cli@latest >/dev/null 2>&1; then
      local go_bin
      go_bin="$(go env GOPATH)/bin/cli"
      if [[ -x "$go_bin" ]] && "$go_bin" --version >/dev/null 2>&1; then
        SUPABASE_CMD="$go_bin"
        export SUPABASE_CMD SUPABASE_PROJECT_DIR
        return 0
      fi
    fi
  fi

  return 1
}

install_supabase_cli() {
  if pick_supabase_cmd; then
    return 0
  fi

  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    echo "Installing Supabase CLI (devDependency) in $SUPABASE_PROJECT_DIR ..."
    if (cd "$SUPABASE_PROJECT_DIR" && npm install --save-dev ${SUPABASE_VERSION:+supabase@${SUPABASE_VERSION}} supabase >/dev/null 2>&1); then
      if pick_supabase_cmd; then
        return 0
      fi
    fi
  fi

  echo "Supabase CLI not runnable without Homebrew. Attempts failed: local devDependency, ephemeral npx, Go install." >&2
  echo "Hints: set SUPABASE_VERSION (e.g. 2.47.2), check network/GitHub access, or install Go." >&2
  exit 1
}
# ----------------------------------------------------------

# Derive a stable project id from the app directory (used by 'supabase stop --project-id')
supabase_project_id() {
  local base="chatbot-ui"
  if [[ -d "$PROJECT_ROOT/chatbot-ui" ]]; then
    base="$(basename "$PROJECT_ROOT/chatbot-ui")"
  fi
  # normalize to alnum/underscore
  echo "${base//[^A-Za-z0-9_]/}"
}

ensure_frontend_env() {
  if [[ "$START_FRONTEND" != "1" ]]; then
    return 0
  fi

  local env_file="$PROJECT_ROOT/chatbot-ui/.env.local"
  if [[ ! -f "$env_file" ]]; then
    echo "[start_env] Creating chatbot-ui/.env.local from template..."
    if [[ -f "$PROJECT_ROOT/chatbot-ui/.env.local.example" ]]; then
      cp "$PROJECT_ROOT/chatbot-ui/.env.local.example" "$env_file"
    else
      touch "$env_file"
    fi
  fi

  local openai_base="${OPENAI_API_BASE_URL:-http://localhost:5000/v1}"
  local openai_key="${OPENAI_API_KEY:-local-placeholder}"

  case "$SUPABASE_MODE" in
    cli)
      # Ensure Docker engine first: use if running; otherwise try Lima VM
      ensure_docker_engine

      install_supabase_cli

      local status_output
      local status_rc
      set +e
      status_output=$(eval "$SUPABASE_CMD status" 2>&1)
      status_rc=$?
      set -e

      if (( status_rc != 0 )) || [[ "$status_output" != *"API URL"* ]]; then
        echo "[start_env] Starting Supabase local stack..."
        # If a previous stack is using ports, stop it cleanly by project-id
        local pid
        pid="$(supabase_project_id)"
        set +e
        eval "$SUPABASE_CMD stop --project-id $pid" >/dev/null 2>&1
        set -e
        eval "$SUPABASE_CMD start"
        status_output=$(eval "$SUPABASE_CMD status" 2>&1)
      fi

      local clean_status
      clean_status=$(printf '%s\n' "$status_output" | sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g')

      local parsed
      parsed=$(printf '%s\n' "$clean_status" | python3 <<'PY'
import re, sys
text = sys.stdin.read()
patterns = {
    "api": r"API URL:\s*(\S+)",
    "anon": r"anon key:\s*([A-Za-z0-9._-]+)",
    "service": r"service role key:\s*([A-Za-z0-9._-]+)",
}
values = []
for name in ("api", "anon", "service"):
    match = re.search(patterns[name], text, re.IGNORECASE)
    values.append(match.group(1).strip() if match else "")
print("|".join(values))
PY
)

      local api_url="${parsed%%|*}"
      local remainder="${parsed#*|}"
      local anon_key="${remainder%%|*}"
      local service_key="${parsed##*|}"

      if [[ -z "$api_url" || -z "$anon_key" || -z "$service_key" ]]; then
        echo "[start_env] Failed to parse Supabase credentials from 'supabase status'." >&2
        echo "$clean_status" >&2
        exit 1
      fi

      set_env_var_in_file "$env_file" "NEXT_PUBLIC_SUPABASE_URL" "$api_url"
      set_env_var_in_file "$env_file" "NEXT_PUBLIC_SUPABASE_ANON_KEY" "$anon_key"
      set_env_var_in_file "$env_file" "SUPABASE_SERVICE_ROLE_KEY" "$service_key"
      ;;
    local)
      local local_env_file="${SUPABASE_LOCAL_ENV_FILE:-$PROJECT_ROOT/chatbot-ui/supabase/local.supabase.env}"
      local local_env_template="${SUPABASE_LOCAL_ENV_TEMPLATE:-${local_env_file}.example}"
      mkdir -p "$(dirname "$local_env_file")"
      if [[ ! -f "$local_env_file" ]]; then
        if [[ -f "$local_env_template" ]]; then
          cp "$local_env_template" "$local_env_file"
        else
          cat > "$local_env_file" <<'EOF'
# Local Supabase credentials go here. Copy values from your locally running Supabase stack.
NEXT_PUBLIC_SUPABASE_URL=http://127.0.0.1:54321
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
EOF
        fi
        echo "[start_env] Created $local_env_file. Fill it with your local Supabase keys." >&2
      fi

      local local_values
      local_values=$(python3 - "$local_env_file" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("||")
    sys.exit(0)

values = {}
for raw in path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    values[key.strip()] = value.strip()

print("|".join([
    values.get("NEXT_PUBLIC_SUPABASE_URL", ""),
    values.get("NEXT_PUBLIC_SUPABASE_ANON_KEY", ""),
    values.get("SUPABASE_SERVICE_ROLE_KEY", ""),
]))
PY
)

      local file_api="${local_values%%|*}"
      local remaining="${local_values#*|}"
      local file_anon="${remaining%%|*}"
      local file_service="${local_values##*|}"

      local api_url="${SUPABASE_URL:-${file_api:-http://127.0.0.1:54321}}"
      local anon_key="${SUPABASE_ANON_KEY:-$file_anon}"
      local service_key="${SUPABASE_SERVICE_ROLE_KEY:-$file_service}"

      set_env_var_in_file "$env_file" "NEXT_PUBLIC_SUPABASE_URL" "$api_url"
      set_env_var_in_file "$env_file" "NEXT_PUBLIC_SUPABASE_ANON_KEY" "$anon_key"
      set_env_var_in_file "$env_file" "SUPABASE_SERVICE_ROLE_KEY" "$service_key"

      if [[ -z "$anon_key" || -z "$service_key" ]]; then
        echo "[start_env] Warning: Supabase keys are blank in $local_env_file. Update the file or export SUPABASE_ANON_KEY/SUPABASE_SERVICE_ROLE_KEY." >&2
      fi
      ;;
    manual)
      if [[ -n "${SUPABASE_URL:-}" ]]; then
        set_env_var_in_file "$env_file" "NEXT_PUBLIC_SUPABASE_URL" "$SUPABASE_URL"
      fi
      if [[ -n "${SUPABASE_ANON_KEY:-}" ]]; then
        set_env_var_in_file "$env_file" "NEXT_PUBLIC_SUPABASE_ANON_KEY" "$SUPABASE_ANON_KEY"
      fi
      if [[ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]]; then
        set_env_var_in_file "$env_file" "SUPABASE_SERVICE_ROLE_KEY" "$SUPABASE_SERVICE_ROLE_KEY"
      fi

      python3 - "$env_file" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
missing = []
if path.exists():
    content = path.read_text().splitlines()
else:
    content = []

values = {}
for raw in content:
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    values[key.strip()] = value.strip()

required = (
    "NEXT_PUBLIC_SUPABASE_URL",
    "NEXT_PUBLIC_SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
)

missing = [key for key in required if not values.get(key)]
if missing:
    print(
        "[start_env] Warning: "
        + ", ".join(missing)
        + " not set in "
        + str(path)
        + ". Provide Supabase credentials or export them before running the frontend.",
        file=sys.stderr,
    )
PY
      ;;
  esac

  set_env_var_in_file "$env_file" "OPENAI_API_BASE_URL" "$openai_base"
  set_env_var_in_file "$env_file" "OPENAI_API_KEY" "$openai_key"

  # shellcheck source=/dev/null
  set -a
  source "$env_file"
  set +a
}

start_backend() {
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
  local host="${CHATBOT_HOST:-0.0.0.0}"
  local port="${CHATBOT_PORT:-3000}"
  export NEXT_TELEMETRY_DISABLED="${NEXT_TELEMETRY_DISABLED:-1}"
  (cd "$PROJECT_ROOT/chatbot-ui" && npm run dev -- --hostname "$host" --port "$port")
}

cleanup() {
  local code=$?
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    if kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
      echo "Stopping frontend (PID $FRONTEND_PID)..."
      kill "$FRONTEND_PID" >/dev/null 2>&1 || true
      wait "$FRONTEND_PID" >/dev/null 2>&1 || true
    fi
  fi
  if [[ -n "${BACKEND_PID:-}" ]]; then
    if kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
      echo "Stopping backend (PID $BACKEND_PID)..."
      kill "$BACKEND_PID" >/dev/null 2>&1 || true
      wait "$BACKEND_PID" >/dev/null 2>&1 || true
    fi
  fi
  return $code
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! check_redis; then
  start_redis
  if ! check_redis; then
    echo "Unable to reach Redis at $REDIS_URL even after attempting to start it." >&2
    exit 1
  fi
fi

echo "Redis available at $REDIS_URL"

install_python_requirements
install_frontend_dependencies
ensure_frontend_env

echo "Launching backend using UI_MODE=$UI_MODE..."
start_backend &
BACKEND_PID=$!
PIDS=("$BACKEND_PID")

if [[ "$START_FRONTEND" == "1" ]]; then
  echo "Launching Chatbot UI frontend..."
  start_frontend &
  FRONTEND_PID=$!
  PIDS+=("$FRONTEND_PID")
fi

echo "Backend PID: $BACKEND_PID"
if [[ "$START_FRONTEND" == "1" ]]; then
  echo "Frontend PID: $FRONTEND_PID"
fi

echo "Press Ctrl+C to stop both services."

set +e
wait "${PIDS[@]}"
exit_code=$?
set -e
exit $exit_code
