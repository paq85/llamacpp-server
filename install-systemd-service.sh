#!/usr/bin/env bash
set -euo pipefail

resolve_script_dir() {
  local source_path="${BASH_SOURCE[0]}"

  while [[ -L "$source_path" ]]; do
    local source_dir

    source_dir="$(cd -P "$(dirname "$source_path")" && pwd)"
    source_path="$(readlink "$source_path")"

    if [[ "$source_path" != /* ]]; then
      source_path="$source_dir/$source_path"
    fi
  done

  cd -P "$(dirname "$source_path")" && pwd
}

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[&|]/\\&/g'
}

ROOT="${ROOT:-$(resolve_script_dir)}"
UNIT_NAME="${UNIT_NAME:-paq-llamacpp-server.service}"
START_NOW=1
RUN_USER="${RUN_USER:-}"
RUN_GROUP="${RUN_GROUP:-}"
RUN_HOME="${RUN_HOME:-}"
TEMP_RENDERED_UNIT=""

cleanup() {
  if [[ -n "$TEMP_RENDERED_UNIT" && -f "$TEMP_RENDERED_UNIT" ]]; then
    rm -f "$TEMP_RENDERED_UNIT"
  fi
}

render_unit() {
  local owner_user="$RUN_USER"
  local owner_group="$RUN_GROUP"
  local owner_home="$RUN_HOME"

  if [[ -z "$owner_user" ]]; then
    owner_user="$(stat -c %U "$ROOT")"
  fi

  if [[ -z "$owner_group" ]]; then
    owner_group="$(stat -c %G "$ROOT")"
  fi

  if [[ -z "$owner_home" ]]; then
    if command -v getent >/dev/null 2>&1; then
      owner_home="$(getent passwd "$owner_user" | cut -d: -f6)"
    fi
  fi

  if [[ -z "$owner_home" ]]; then
    owner_home="$(eval printf '%s' "~$owner_user")"
  fi

  if [[ -z "$owner_user" || -z "$owner_group" || -z "$owner_home" || "$owner_home" == "~$owner_user" ]]; then
    echo "Could not determine RUN_USER/RUN_GROUP/RUN_HOME for rendering $UNIT_NAME" >&2
    exit 1
  fi

  local escaped_root escaped_user escaped_group escaped_home

  escaped_root="$(escape_sed_replacement "$ROOT")"
  escaped_user="$(escape_sed_replacement "$owner_user")"
  escaped_group="$(escape_sed_replacement "$owner_group")"
  escaped_home="$(escape_sed_replacement "$owner_home")"

  TEMP_RENDERED_UNIT="$(mktemp)"

  sed \
    -e "s|@ROOT@|$escaped_root|g" \
    -e "s|@RUN_USER@|$escaped_user|g" \
    -e "s|@RUN_GROUP@|$escaped_group|g" \
    -e "s|@RUN_HOME@|$escaped_home|g" \
    "$UNIT_SOURCE" > "$TEMP_RENDERED_UNIT"
}

trap cleanup EXIT

usage() {
  cat <<EOF
Usage: sudo ./install-systemd-service.sh [--no-start] [--unit UNIT]

Installs a rendered systemd unit into /etc/systemd/system, reloads systemd,
enables the service for boot, and starts or restarts it immediately unless
--no-start is provided.

Options:
  --no-start     Install and enable the unit without starting it
  --unit UNIT    Unit file in the repo root to render/install (default: $UNIT_NAME)
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --no-start)
      START_NOW=0
      ;;
    --unit)
      shift

      if (( $# == 0 )); then
        echo "Missing value for --unit" >&2
        usage >&2
        exit 2
      fi

      UNIT_NAME="$1"
      ;;
    --unit=*)
      UNIT_NAME="${1#*=}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

UNIT_SOURCE="${UNIT_SOURCE:-$ROOT/$UNIT_NAME}"
UNIT_DEST="${UNIT_DEST:-/etc/systemd/system/$UNIT_NAME}"

if (( EUID != 0 )); then
  echo "This installer must run as root. Try: sudo ./install-systemd-service.sh" >&2
  exit 1
fi

for command_name in install mktemp sed stat systemctl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

if [[ ! -f "$UNIT_SOURCE" ]]; then
  echo "Unit file not found: $UNIT_SOURCE" >&2
  exit 1
fi

render_unit

install -m 0644 "$TEMP_RENDERED_UNIT" "$UNIT_DEST"
echo "Installed rendered $UNIT_NAME to $UNIT_DEST"

if command -v systemd-analyze >/dev/null 2>&1; then
  echo "Verifying unit with systemd-analyze..."
  systemd-analyze verify "$UNIT_DEST"
fi

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling $UNIT_NAME for boot..."
systemctl enable "$UNIT_NAME" >/dev/null

if (( START_NOW )); then
  if systemctl is-active --quiet "$UNIT_NAME"; then
    echo "Restarting active service $UNIT_NAME..."
    systemctl restart "$UNIT_NAME"
  else
    echo "Starting service $UNIT_NAME..."
    systemctl start "$UNIT_NAME"
  fi

  echo
  systemctl --no-pager --full status "$UNIT_NAME" || true
else
  echo "Skipping immediate start (--no-start)."
  echo "Start it later with: sudo systemctl start $UNIT_NAME"
fi
