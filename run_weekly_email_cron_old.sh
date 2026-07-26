#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$ROOT_DIR/AutomationDetection/bin/activate"
CONFIG_PATH="${1:-$ROOT_DIR/weekly_email.env}"

EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  shift
  EXTRA_ARGS=("$@")
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config file: $CONFIG_PATH"
  echo "Create it from: $ROOT_DIR/weekly_email.env.example"
  exit 1
fi

if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
fi

# Load KEY=VALUE pairs from config.
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line#${raw_line%%[![:space:]]*}}"
  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue
  if [[ "$line" != *=* ]]; then
    continue
  fi

  key="${line%%=*}"
  value="${line#*=}"

  # Trim whitespace around key and value.
  key="${key%${key##*[![:space:]]}}"
  key="${key#${key%%[![:space:]]*}}"
  value="${value#${value%%[![:space:]]*}}"
  value="${value%${value##*[![:space:]]}}"

  # If user already quoted the value, strip one pair of matching quotes.
  if [[ ${#value} -ge 2 ]]; then
    if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
      value="${value:1:${#value}-2}"
    fi
  fi

  export "$key=$value"
done < "$CONFIG_PATH"

require_var() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "Required variable is empty: $var_name"
    exit 1
  fi
}

require_var SMTP_HOST
require_var SMTP_PORT
require_var SMTP_USER
require_var SMTP_PASSWORD
require_var ALERT_EMAIL_FROM
require_var ALERT_EMAIL_TO

FLARE_THRESHOLD_MULTIPLIER="${FLARE_THRESHOLD_MULTIPLIER:-2.0}"
CONSECUTIVE_POINTS="${CONSECUTIVE_POINTS:-3}"
INCREMENTAL_PERCENT="${INCREMENTAL_PERCENT:-0.3}"
EMAIL_SUBJECT_PREFIX="${EMAIL_SUBJECT_PREFIX:-AutomationDetection weekly flare alert}"
EMAIL_FORCE_SEND="${EMAIL_FORCE_SEND:-0}"
SOURCE_NAME="${SOURCE_NAME:-}"

export SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASSWORD ALERT_EMAIL_FROM ALERT_EMAIL_TO

CMD=(
  python "$ROOT_DIR/AutomatedScript.py"
  --email-on-detections
  --smtp-host "$SMTP_HOST"
  --smtp-port "$SMTP_PORT"
  --smtp-user "$SMTP_USER"
  --email-from "$ALERT_EMAIL_FROM"
  --email-to "$ALERT_EMAIL_TO"
  --email-subject-prefix "$EMAIL_SUBJECT_PREFIX"
  --incremental-percent "$INCREMENTAL_PERCENT"
  --flare-threshold-multiplier "$FLARE_THRESHOLD_MULTIPLIER"
  --consecutive-points "$CONSECUTIVE_POINTS"
)

if [[ -n "$SOURCE_NAME" ]]; then
  CMD+=(--source "$SOURCE_NAME")
fi

if [[ "$EMAIL_FORCE_SEND" == "1" ]]; then
  CMD+=(--email-force-send)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

unset SMTP_PASSWORD
