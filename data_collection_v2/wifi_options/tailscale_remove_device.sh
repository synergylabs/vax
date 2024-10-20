#!/bin/bash

tailnet="prasoonpatidar.github"
# Replace with a path to your Tailscale API key.
apikey=`cat /home/vax/.ssh/tailscale_api.key`
# echo "$apikey"
if [ -z "$*" ]; then echo "No device id found, Please provide device id to remove as an argument"; exit; fi

targetname=$1
curl -s "https://api.tailscale.com/api/v2/tailnet/$tailnet/devices" -u "$apikey:" |jq -r '.devices[] |  "\(.id) \(.name)"' |
  while read id name; do
    if [[ $name = *"$targetname"* ]]
    then
      echo $name $id " includes " $name " in its name - getting rid of it"
      curl -s -X DELETE "https://api.tailscale.com/api/v2/device/$id" -u "$apikey:"
    else
      echo $name" does not have that string in its name, keeping it"
    fi
  done
