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
export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
export TORCHELASTIC_MAX_RESTARTS=0

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
echo "TRAINING ${MODE} model ${MODEL_NAME}"
cd ${PROJECT_DIR}
export MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
fi
mkdir ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/.git/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/tools/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/scripts/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/pysgg/ ${MODEL_DIRNAME} &&


export OMP_NUM_THREADS=1


python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=$NUM_GPUS \
       tools/relation_train_net.py \
       --config-file "configs/e2e_relBGNN_vg.yaml" \
       DEBUG False\
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
       MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR 0.13 \
       MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE 1.6 \
       EXPERIMENT_NAME "$MODEL_NAME" \
       SOLVER.IMS_PER_BATCH $[$BATCH_SIZE*$NUM_GPUS] \
       SOLVER.PRE_VAL ${PRE_VAL} \
       SOLVER.BASE_LR ${LR} \
       SOLVER.MAX_ITER ${MAX_ITER} \
       TEST.IMS_PER_BATCH ${NUM_GPUS} \
       SOLVER.VAL_PERIOD 2000 \
       SOLVER.CHECKPOINT_PERIOD 2000 \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${USE_GT_BOX} \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${USE_GT_OBJECT_LABEL} 2>&1 | tee ${MODEL_DIRNAME}/log_train.log &&
echo "Finished training ${MODE} model ${MODEL_NAME}" || echo "Failed to train ${MODE} model ${MODEL_NAME}"
