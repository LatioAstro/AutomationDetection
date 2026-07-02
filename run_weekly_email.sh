#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$ROOT_DIR/AutomationDetection/bin/activate"

FORCE_SEND=0
if [[ "${1:-}" == "--force-send" ]]; then
  FORCE_SEND=1
fi

if [[ -f "$VENV_ACTIVATE" ]]; then
  # Activate project virtual environment if available.
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
fi

read -rp "Gmail address (sender): " GMAIL_FROM
if [[ -z "$GMAIL_FROM" ]]; then
  echo "Sender email is required."
  exit 1
fi

read -rp "Recipients (comma-separated) [${GMAIL_FROM}]: " ALERT_TO
ALERT_TO="${ALERT_TO:-$GMAIL_FROM}"

read -rp "Analyze one source only? (leave blank = all NameCSV): " SOURCE_NAME
read -rp "Flare threshold multiplier [2.0]: " FLARE_MULTIPLIER
FLARE_MULTIPLIER="${FLARE_MULTIPLIER:-2.0}"

read -rp "Consecutive points [3]: " CONSECUTIVE_POINTS
CONSECUTIVE_POINTS="${CONSECUTIVE_POINTS:-3}"

read -rsp "Gmail App Password (16 chars): " SMTP_PASSWORD
echo
if [[ -z "$SMTP_PASSWORD" ]]; then
  echo "App password is required."
  exit 1
fi

export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="$GMAIL_FROM"
export SMTP_PASSWORD
export ALERT_EMAIL_FROM="$GMAIL_FROM"
export ALERT_EMAIL_TO="$ALERT_TO"

CMD=(
  python "$ROOT_DIR/AutomatedScript.py"
  --email-on-detections
  --flare-threshold-multiplier "$FLARE_MULTIPLIER"
  --consecutive-points "$CONSECUTIVE_POINTS"
)

if [[ -n "$SOURCE_NAME" ]]; then
  CMD+=(--source "$SOURCE_NAME")
fi

if [[ "$FORCE_SEND" -eq 1 ]]; then
  CMD+=(--email-force-send)
fi

echo "Running incremental scan and weekly email check..."
"${CMD[@]}"

# Clear sensitive value from current shell process.
unset SMTP_PASSWORD

echo "Done."
