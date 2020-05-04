#!/usr/bin/env bash

set -e

PROJECT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" && pwd  )"
PROJECT_ROOT_DIR="$PROJECT_SCRIPT_DIR/.."

# This is for solving the python double free or corruption error
export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4":$LD_PRELOAD

if [ ! -z $ARNOLD_TRIAL_ID ]; then
  TASK_NAME=$(echo $ARNOLD_TASK_NAME | sed -r 's/\s/_/g')
  TRAIN_DIR=$ENV_NEW_WJ_DIR/vc/$ARNOLD_TASK_OWNER/arnold/classification/$TASK_NAME/$ARNOLD_TRIAL_ID
  echo $TRAIN_DIR
else
  TRAIN_DIR=$PROJECT_ROOT_DIR/logs
fi


if [ ! -d $TRAIN_DIR  ];then
  echo "Train dir not exist, mkdir: " $TRAIN_DIR
  mkdir -p $TRAIN_DIR
fi

cd $PROJECT_ROOT_DIR

python3 train.py \
  --ckpt=$TRAIN_DIR \
  --log_path=$TRAIN_DIR \
  $@