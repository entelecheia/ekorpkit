#!/bin/bash
chezmoi update --apply=false
chezmoi init --apply --verbose
chezmoi apply

echo "Get ngrok tunnel url..."
WEBHOOK_BASE_URL=$(bash /scripts/ngrok_url.sh)
export WEBHOOK_BASE_URL
echo "WEBHOOK_BASE_URL=$WEBHOOK_BASE_URL"
