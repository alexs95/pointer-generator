#!/bin/sh
#SBATCH -p GpuQ
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH -A project
#SBATCH --mail-user=email@domain.com
#SBATCH --mail-type=BEGIN,END

module load cuda/11.2
module load conda/2

conda init
source activate summarization3.6

cd $SLURM_SUBMIT_DIR
time python run_summarization.py \
--mode=decode \
--data_path='../cnn-dailymail/finished_files/chunked/test_*' \
--vocab_path=../data/finished_files/vocab \
--log_root=../pointer-generator/checkpoint \
--exp_name=pretrained_model_tf1.2.1 \
--max_enc_steps=400 \
--max_dec_steps=120 \
--min_dec_steps=35 \
--beam_size=4 \
--single_pass=1 \
--coverage=1
