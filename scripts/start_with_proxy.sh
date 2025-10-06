#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$SCRIPT_DIR/supabase-proxy.conf.template"
RUNTIME_DIR="$PROJECT_ROOT/.tmp/supabase-proxy"
CONFIG="$RUNTIME_DIR/nginx.conf"
LOG_DIR="$RUNTIME_DIR/logs"

LISTEN_ADDR="${SUPABASE_PROXY_LISTEN:-0.0.0.0}"
EXTERNAL_HOST="${SUPABASE_PROXY_HOST:-}"
FRONTEND_PORT="${SUPABASE_PROXY_FRONTEND_PORT:-80}"

if [[ $# -ge 1 && -n "${1:-}" ]]; then
  EXTERNAL_HOST="$1"
fi

if [[ $# -ge 2 && -n "${2:-}" ]]; then
  LISTEN_ADDR="$2"
fi

if [[ $# -ge 3 && -n "${3:-}" ]]; then
  FRONTEND_PORT="$3"
fi

if [[ -z "$EXTERNAL_HOST" ]]; then
  if [[ "$LISTEN_ADDR" == "0.0.0.0" ]]; then
    EXTERNAL_HOST="127.0.0.1"
  else
    EXTERNAL_HOST="$LISTEN_ADDR"
  fi
fi

if [[ ! -f "$TEMPLATE" ]]; then
  echo "[proxy] Template not found at $TEMPLATE" >&2
  exit 1
fi

if ! command -v nginx >/dev/null 2>&1; then
  echo "[proxy] nginx binary not found on PATH. Install nginx (e.g. via Homebrew: brew install nginx)." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

HOST_FOR_ENV="$EXTERNAL_HOST"

python3 - "$TEMPLATE" "$CONFIG" "$LISTEN_ADDR" "$FRONTEND_PORT" <<'PY'
import socket
import sys
from pathlib import Path

template_path, config_path, listen_addr, frontend_port = sys.argv[1:]

def port_available(addr: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bind_addr = addr
    if addr in ('', '0.0.0.0'):
        bind_addr = ''
    try:
        sock.bind((bind_addr, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True

def build_block(port: int, label: str) -> str:
    if port_available(listen_addr, port):
        print(f"[proxy] Port {port} available on {listen_addr}; enabling {label} proxy.")
        return (
            f"  server {{\n"
            f"    listen {listen_addr}:{port};\n"
            f"    server_name _;\n\n"
            f"    location / {{\n"
            f"      proxy_pass http://127.0.0.1:{port};\n"
            f"    }}\n"
            f"  }}\n"
        )
    print(f"[proxy] Port {port} already in use on {listen_addr}; skipping {label} proxy.")
    return f"  # Skipping {label} proxy on {listen_addr}:{port} (already bound)\n"

content = Path(template_path).read_text()
content = content.replace("__SUPABASE_PROXY_LISTEN__", listen_addr)
content = content.replace("__SUPABASE_FRONTEND_PORT__", frontend_port)
content = content.replace("__SUPABASE_PROXY_REST_BLOCK__", build_block(54321, "Supabase REST"))
content = content.replace("__SUPABASE_PROXY_STUDIO_BLOCK__", build_block(54323, "Supabase Studio"))
content = content.replace("__SUPABASE_PROXY_MAILPIT_BLOCK__", build_block(54324, "Mailpit"))
Path(config_path).write_text(content)
PY

echo "[proxy] Generated nginx config at $CONFIG"
echo "[proxy] Listening on $LISTEN_ADDR for Supabase endpoints (REST 54321, Studio 54323, Mailpit 54324)"
echo "[proxy] Requests will be proxied to 127.0.0.1"
echo "[proxy] External host advertised to runman.sh: $HOST_FOR_ENV"
echo "[proxy] Frontend traffic will be served on port $FRONTEND_PORT (HTTP redirect to http://$HOST_FOR_ENV:3000)"

if [[ -f "$LOG_DIR/nginx.pid" ]]; then
  if nginx -c "$CONFIG" -p "$RUNTIME_DIR" -s stop 2>/dev/null; then
    sleep 1
  fi
fi

cleanup() {
  if [[ -f "$LOG_DIR/nginx.pid" ]]; then
    nginx -c "$CONFIG" -p "$RUNTIME_DIR" -s stop >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

nginx -c "$CONFIG" -p "$RUNTIME_DIR"

pid_msg="unknown"
if [[ -f "$LOG_DIR/nginx.pid" ]]; then
  pid_msg="$(cat "$LOG_DIR/nginx.pid")"
fi

echo "[proxy] nginx started (pid $pid_msg)"
echo "[proxy] Launching runman.sh with SUPABASE_EXTERNAL_HOST=$HOST_FOR_ENV"

export SUPABASE_EXTERNAL_HOST="$HOST_FOR_ENV"
SUPABASE_EXTERNAL_HOST="$HOST_FOR_ENV" "$SCRIPT_DIR/runman.sh"
