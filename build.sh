#!/bin/bash
CONTAINER_NAME='suspension_of_disbelief'
DATE=$(date '+%F')
cd images/base
docker build -t $CONTAINER_NAME:latest .