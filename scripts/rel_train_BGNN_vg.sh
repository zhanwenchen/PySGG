#!/bin/bash
timestamp() {
  date +"%Y-%m-%d %T"
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
echo "TRAINING PredCls model ${MODEL_NAME}"
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
       --config-file ${CONFIG_FILE} \
       DEBUG False\
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
       EXPERIMENT_NAME "$MODEL_NAME" \
       SOLVER.IMS_PER_BATCH $[$BATCH_SIZE*$NUM_GPUS] \
       SOLVER.PRE_VAL True \
       SOLVER.BASE_LR ${LR} \
       TEST.IMS_PER_BATCH ${NUM_GPUS} \
       SOLVER.VAL_PERIOD 2000 \
       SOLVER.CHECKPOINT_PERIOD 2000 \
       MODEL.ROI_RELATION_HEAD.PAIRWISE.USING_EXPLICIT_PAIRWISE ${USING_EXPLICIT_PAIRWISE} \
       MODEL.ROI_RELATION_HEAD.PAIRWISE.EXPLICIT_PAIRWISE_DATA ${EXPLICIT_PAIRWISE_DATA} \
       MODEL.ROI_RELATION_HEAD.PAIRWISE.EXPLICIT_PAIRWISE_FUNC ${EXPLICIT_PAIRWISE_FUNC} \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${USE_GT_BOX} \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${USE_GT_OBJECT_LABEL} 2>&1 | tee ${MODEL_DIRNAME}/log_train.log &&
echo "Finished training PredCls model ${MODEL_NAME}" || echo "Failed to train PredCls model ${MODEL_NAME}"
