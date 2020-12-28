#!/usr/bin/env bash

if [[ "$1" == "" ]]; then
  echo "Please provide name of the job."
  exit 1
else
  export JOB_NAME="$1"
fi

export CONFIG_NAME="$2"

export TRAINING_NAME="$3"

envsubst < command.sh

. command.sh

envsubst < job.yml > /dev/null 2>&1
