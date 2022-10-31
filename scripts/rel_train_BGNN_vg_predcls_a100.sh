#!/bin/bash

#SBATCH -A sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4 # need to match number of gpus
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB # need to match batch size.
#SBATCH -J bgnn_pairwise_hadamard_mha_explicit_only_specific_config_4GPU_riv_1_predcls # TODO: CHANGE THIS
#SBATCH -o /home/pct4et/pysgg/log/%x-%A.out
#SBATCH -e /home/pct4et/pysgg/log/%x-%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pct4et@virginia.edu
#SBACTH --constraint=a100_80gb

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

export PROJECT_DIR=${HOME}/pysgg
export MODEL_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export LOGDIR=${PROJECT_DIR}/log
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/

if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
else
  module purge
  module load singularity

  export SEED=1234
  export BATCH_SIZE=24
  export MAX_ITER=50000
  export LR=2e-3
  export USE_GSC_FE=False
  export SINGULARITYENV_PREPEND_PATH="${HOME}/.conda/envs/pysgg/bin:/opt/conda/condabin"
  export CONFIG_FILE=configs/e2e_relBGNN_vg_predcls.yaml
  export NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  export DATA_DIR_VG_RCNN=/project/sds-rise/zhanwen/datasets
  export USE_GT_BOX=True
  export USE_GT_OBJECT_LABEL=True
  export USING_EXPLICIT_PAIRWISE_ONLY=True
  export USING_EXPLICIT_PAIRWISE=True
  export EXPLICIT_PAIRWISE_DATA='hadamard'
  export EXPLICIT_PAIRWISE_FUNC='mha'

  singularity exec --nv --env LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:${HOME}/.conda/envs/pysgg/lib" docker://pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel ${PROJECT_DIR}/scripts/rel_train_BGNN_vg.sh
fi
