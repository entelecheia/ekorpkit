#!/bin/sh

NGROK_DOCKER_PORT=${NGROK_DOCKER_PORT:-4040}

echo_url() {
  ngrok_tunnels=$(curl -s "http://ngrok:${NGROK_DOCKER_PORT}/api/tunnels")
  index=$(echo "$ngrok_tunnels" | jq '.tunnels | map(.proto == "https") | index(true)')
  echo "$ngrok_tunnels" | jq -r ".tunnels[$index].public_url"
}

n=0
until [ "$n" -ge 5 ]
do
  echo_url && break
  n=$((n+1))
  sleep 5
done
