#!/usr/bin/env bash
set -euo pipefail

SERVICE_DEFAULT="paq-llamacpp-server.service"
CMD="${1:-help}"
SERVICE="${2:-$SERVICE_DEFAULT}"

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found on this host; cannot manage systemd units." >&2
  exit 1
fi

run_systemctl() {
  if [[ $EUID -ne 0 ]]; then
    sudo systemctl "$@"
  else
    systemctl "$@"
  fi
}

run_status() {
  systemctl status --no-pager "$1"
}

run_journal() {
  journalctl -u "$1" --no-pager
}

run_tail() {
  journalctl -u "$1" -f
}

print_help() {
  cat <<'EOF'
servicectl.sh - simple wrapper to manage the paq-llamacpp-server.service

Usage:
  scripts/servicectl.sh <command> [service]

Commands:
  start        Reset failed state then start the service
  start-now    Enable and start the service (enable --now)
  stop         Stop the service
  restart      Reset failed state then restart the service
  status       Show systemctl status --no-pager
  enable       Enable the service at boot
  enable-now   Enable and start the service
  disable      Disable the service
  mask         Mask the service
  unmask       Unmask the service
  reset-failed Clear failed state for the unit
  journal      Show journalctl for the unit (no pager)
  tail         Follow journalctl logs for the unit
  help         Show this help

Examples:
  scripts/servicectl.sh start
  scripts/servicectl.sh status paq-llamacpp-server.service
  scripts/servicectl.sh start-now

By default the service used is 'paq-llamacpp-server.service'.
EOF
}

case "$CMD" in
  start)
    run_systemctl reset-failed "$SERVICE" || true
    run_systemctl start "$SERVICE"
    ;;
  start-now)
    run_systemctl reset-failed "$SERVICE" || true
    run_systemctl enable --now "$SERVICE"
    ;;
  stop)
    run_systemctl stop "$SERVICE"
    ;;
  restart)
    run_systemctl reset-failed "$SERVICE" || true
    run_systemctl restart "$SERVICE"
    ;;
  status)
    run_status "$SERVICE"
    ;;
  enable)
    run_systemctl enable "$SERVICE"
    ;;
  enable-now)
    run_systemctl enable --now "$SERVICE"
    ;;
  disable)
    run_systemctl disable "$SERVICE"
    ;;
  mask)
    run_systemctl mask "$SERVICE"
    ;;
  unmask)
    run_systemctl unmask "$SERVICE"
    ;;
  reset-failed)
    run_systemctl reset-failed "$SERVICE"
    ;;
  journal)
    run_journal "$SERVICE"
    ;;
  tail)
    run_tail "$SERVICE"
    ;;
  help|--help|-h)
    print_help
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    print_help
    exit 1
    ;;
esac
