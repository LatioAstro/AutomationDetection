#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${1:-}"

if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
    if [[ -f "$ROOT_DIR/PythonFiles/env_loader.py" ]]; then
        while IFS= read -r line; do
            if [[ -n "$line" ]]; then
                eval "$line"
            fi
        done < <(python3 "$ROOT_DIR/PythonFiles/env_loader.py" "$ENV_FILE")
    else
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    fi
    shift
fi

VENV_ACTIVATE="$ROOT_DIR/AutomationDetection/bin/activate"

###############################################################################
# Activate local virtual environment if it exists
###############################################################################

if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_ACTIVATE"
fi

###############################################################################
# Default values
###############################################################################

SOURCE_NAME="${SOURCE_NAME:-}"
FLARE_MULTIPLIER="${FLARE_MULTIPLIER:-${FLARE_THRESHOLD_MULTIPLIER:-2.0}}"
CONSECUTIVE_POINTS="${CONSECUTIVE_POINTS:-3}"

EMAIL_FORCE_SEND="${EMAIL_FORCE_SEND:-0}"
EMAIL_INCLUDE_POTENTIAL_PLOTS="${EMAIL_INCLUDE_POTENTIAL_PLOTS:-0}"
EMAIL_INCLUDE_CONFIRMED_PLOTS="${EMAIL_INCLUDE_CONFIRMED_PLOTS:-0}"

###############################################################################
# Interactive prompts ONLY if variables are missing
###############################################################################

GMAIL_FROM="${SMTP_USER:-}"

if [[ -z "$GMAIL_FROM" ]]; then
    read -rp "Gmail address (sender): " GMAIL_FROM
fi

if [[ -z "$GMAIL_FROM" ]]; then
    echo "Sender email is required."
    exit 1
fi

ALERT_TO="${ALERT_EMAIL_TO:-}"

if [[ -z "$ALERT_TO" ]]; then
    read -rp "Recipients (comma-separated) [${GMAIL_FROM}]: " ALERT_TO
    ALERT_TO="${ALERT_TO:-$GMAIL_FROM}"
fi

SMTP_PASSWORD="${SMTP_PASSWORD:-}"

if [[ -z "$SMTP_PASSWORD" ]]; then
    read -rsp "Gmail App Password (16 chars): " SMTP_PASSWORD
    echo
fi

if [[ -z "$SMTP_PASSWORD" ]]; then
    echo "App password is required."
    exit 1
fi

###############################################################################
# Export email configuration
###############################################################################

export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="$GMAIL_FROM"
export SMTP_PASSWORD
export ALERT_EMAIL_FROM="$GMAIL_FROM"
export ALERT_EMAIL_TO="$ALERT_TO"

###############################################################################
# Build command
###############################################################################

CMD=(
    python "$ROOT_DIR/AutomatedScript.py"
    --email-on-detections
    --flare-threshold-multiplier "$FLARE_MULTIPLIER"
    --consecutive-points "$CONSECUTIVE_POINTS"
)

if [[ -n "$SOURCE_NAME" ]]; then
    CMD+=(--source "$SOURCE_NAME")
fi

if [[ "$EMAIL_FORCE_SEND" == "1" ]]; then
    CMD+=(--email-force-send)
fi

if [[ "$EMAIL_INCLUDE_POTENTIAL_PLOTS" == "1" ]]; then
    CMD+=(--email-include-potential-plots)
fi

if [[ "$EMAIL_INCLUDE_CONFIRMED_PLOTS" == "1" ]]; then
    CMD+=(--email-include-confirmed-plots)
fi

###############################################################################
# Run
###############################################################################

echo "Running automated weekly detection..."

"${CMD[@]}"

unset SMTP_PASSWORD

echo "Done."