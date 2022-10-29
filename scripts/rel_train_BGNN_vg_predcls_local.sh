#!/bin/bash
timestamp() {
  date +"%Y%m%d%H%M%S"
}

SLURM_JOB_NAME=pairwise_predcls_1GPU_legion_dev
SLURM_JOB_ID=$(timestamp)

error_exit()
{
#   ----------------------------------------------------------------
#   Function for exit due to fatal program error
#       Accepts 1 argument:
#           string containing descriptive error message
#   Source: http://linuxcommand.org/lc3_wss0140.php
#   ----------------------------------------------------------------
    echo "$(timestamp) ERROR ${PROGNAME}: ${1:-"Unknown Error"}" 1>&2
    echo "$(timestamp) ERROR ${PROGNAME}: Exiting Early."
    exit 1
}


export PROJECT_DIR=${HOME}/pysgg
export MODEL_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export LOGDIR=${PROJECT_DIR}/log
export MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/

if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
else
  export CUDA_VISIBLE_DEVICES=0
  export SEED=1234
  export BATCH_SIZE=12
  export MAX_ITER=50000
  export LR=8e-3
  export DATA_DIR_VG_RCNN=${HOME}/datasets
  export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c); ((NUM_GPUS++))
  export USE_GT_BOX=True
  export USE_GT_OBJECT_LABEL=True
  export PRE_VAL=False
  ${PROJECT_DIR}/scripts/rel_train_BGNN_vg.sh
fi
