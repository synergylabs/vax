#!/bin/bash
# Define variables
IMAGE_NAME="ytyou/ticktock:latest-grafana"
CONTAINER_NAME="ticktock"
docker stop $CONTAINER_NAME || true && docker rm $CONTAINER_NAME || true
docker run --name $CONTAINER_NAME -p 3000:3000 -p 6181-6182:6181-6182 -p 6181:6181/udp  $IMAGE_NAME
