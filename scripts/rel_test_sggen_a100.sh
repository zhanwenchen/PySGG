#!/bin/bash

#SBATCH -A sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1 # need to match number of gpus
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB # need to match batch size.
#SBATCH -J test-bgnn_pairwise_hadamard_mha_4GPU_riv_1_predcls-0002000 # TODO: CHANGE THIS
#SBATCH -o /home/pct4et/pysgg/log/%x-%A.out
#SBATCH -e /home/pct4et/pysgg/log/%x-%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pct4et@virginia.edu

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
IFS=- read str1 MODEL_NAME ITER <<< "${SLURM_JOB_NAME}"


export PROJECT_DIR=${HOME}/pysgg
export LOGDIR=${PROJECT_DIR}/log
export MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/

if [! -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} does not exist." 2>&1 | tee -a ${LOGDIR}/${MODEL_NAME}.log
else
  module purge
  module load singularity

  export SINGULARITYENV_PREPEND_PATH="${HOME}/.conda/envs/pysgg/bin:/opt/conda/condabin"
  # export CONFIG_FILE=configs/e2e_relation_X_101_32_8_FPN_1x_pairwise.yaml
  export NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  export DATA_DIR_VG_RCNN=/project/sds-rise/zhanwen/datasets
  export USE_GT_BOX=False
  export USE_GT_OBJECT_LABEL=False

  singularity exec --nv --env LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:${HOME}/.conda/envs/pysgg/lib" docker://pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel ${PROJECT_DIR}/scripts/rel_test.sh
fi
