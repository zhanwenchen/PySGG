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
export TORCHELASTIC_MAX_RESTARTS=0
echo "TRAINING PredCls model ${MODEL_NAME}"
cd ${PROJECT_DIR}
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
fi
mkdir ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/.git/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/tools/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/scripts/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/pysgg/ ${MODEL_DIRNAME} &&


export OMP_NUM_THREADS=1


python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$NUM_GPUS \
       tools/relation_train_net.py \
       --config-file "configs/e2e_relBGNN_vg.yaml" \
       DEBUG False\
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
       MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR 0.13 \
       MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE 1.6 \
       EXPERIMENT_NAME "$MODEL_NAME" \
       SOLVER.IMS_PER_BATCH $[$BATCH_SIZE*$NUM_GPUS] \
       SOLVER.PRE_VAL True \
       TEST.IMS_PER_BATCH $[$NUM_GPUS] \
       SOLVER.VAL_PERIOD 2000 \
       SOLVER.CHECKPOINT_PERIOD 2000 \
       MODEL.ROI_RELATION_HEAD.PAIRWISE.EXPLICIT_PAIRWISE_DATA 'hadamard' \
       MODEL.ROI_RELATION_HEAD.PAIRWISE.EXPLICIT_PAIRWISE_FUNC 'mha' \
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${USE_GT_BOX} \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${USE_GT_OBJECT_LABEL}

       # MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/detection/pretrained_faster_rcnn/model_final.pth \
echo "Finished training PredCls model ${MODEL_NAME}" || echo "Failed to train PredCls model ${MODEL_NAME}"
