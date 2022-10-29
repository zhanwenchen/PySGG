#!/bin/bash
timestamp() {
  date +"%Y%m%d%H%M%S"
}

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

error_check()
{
#   ----------------------------------------------------------------
#   This function simply checks a passed return code and if it is
#   non-zero it returns 1 (error) to the calling script.  This was
#   really only created because I needed a method to store STDERR in
#   other scripts and also check $? but also leave open the ability
#   to add in other stuff, too.
#
#   Accepts 1 arguments:
#       return code from prior command, usually $?
#  ----------------------------------------------------------------
    TO_CHECK=${1:-0}

    if [ "$TO_CHECK" != '0' ]; then
        return 1
    fi

}

get_mode()
{
  if [[ ${USE_GT_BOX} == "True" ]] && [[ ${USE_GT_OBJECT_LABEL} == "True" ]]; then
    export MODE="predcls"
  elif [[ ${USE_GT_BOX} == "True" ]] && [[ ${USE_GT_OBJECT_LABEL} == "False" ]]; then
    export MODE="sgcls"
  elif [[ ${USE_GT_BOX} == "False" ]] && [[ ${USE_GT_OBJECT_LABEL} == "False" ]]; then
    export MODE="sgdet"
  else
    error_exit "Illegal USE_GT_BOX=${USE_GT_BOX} and USE_GT_OBJECT_LABEL=${USE_GT_OBJECT_LABEL} provided."
  fi
}

export MODE=$(get_mode)

IFS=- read str1 MODEL_NAME ITER <<< "${SLURM_JOB_NAME}"

export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo "TRAINING ${MODE} model ${MODEL_NAME}"
python -m torch.distributed.launch --master_port $PORT --nproc_per_node=$NUM_GPUS  \
  tools/relation_test_net.py \
  --config-file "${MODEL_NAME}/config.yml"\
  TEST.IMS_PER_BATCH $[$gpu_num] \
  MODEL.WEIGHT  "${MODEL_NAME}/model_${ITER}.pth"\
  MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
  DATASETS.TEST "('VG_stanford_filtered_with_attribute_test', )" 2>&1 | tee ${MODEL_DIRNAME}/log_test.log &&
echo "Finished testing ${MODE} model ${MODEL_NAME}" || echo "Failed to test ${MODE} model ${MODEL_NAME}"
