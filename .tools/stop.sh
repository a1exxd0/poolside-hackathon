#!/usr/bin/env bash
# Stop the local vLLM serve and the cloudflared quick tunnel started by serve.sh.
set -uo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for name in cloudflared vllm; do
  pf="$TOOLS/$name.pid"
  if [ ! -f "$pf" ]; then echo "no pid file for $name"; continue; fi
  pid="$(cat "$pf")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" && echo "stopped $name (pid $pid)"
    sleep 2
    kill -0 "$pid" 2>/dev/null && { kill -9 "$pid"; echo "  force-killed $name"; }
  else
    echo "$name not running (stale pid $pid)"
  fi
  rm -f "$pf"
done
