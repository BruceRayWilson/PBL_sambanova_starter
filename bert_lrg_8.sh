#! /bin/bash
set -e
#######################
# Edit these variables.
#######################
export OMP_NUM_THREADS=1
MODEL_NAME="bertlg8"
#######################
# Start script timer
SECONDS=0
# Temp file location
DIRECTORY=$$
OUTDIR=${HOME}/${DIRECTORY}

source /opt/sambaflow/venv/bin/activate
cd ${HOME}
echo "Model: ${MODEL_NAME}"
echo "Date: " $(date +%m/%d/%y)
echo "Time: " $(date +%H:%M)

echo "Machine State Before: "
/opt/sambaflow/bin/snfadm -l inventory
echo "COMPILE"
COMMAND="python /opt/sambaflow/apps/nlp/pytorch/transformers_on_rdu/transformers_hook.py compile --model_name_or_path bert-large-uncased --tokenizer_name bert-large-uncased --module_name mlm_ns --task_name mlm_ns --do_eval --max_seq_length 128 --per_device_train_batch_size 256 --per_device_eval_batch_size 256 -b 256 --output_dir=${OUTDIR}/hf_output --overwrite_output_dir --cache_dir ${OUTDIR}/cache --compiler-configs-file /opt/sambaflow/apps/nlp/pytorch/transformers_on_rdu/human_decisions/compiler_configs/compiler_configs_bertlarge_sc_mlm_ml_perf_fullfeature_macv2_clipping.json --mac-human-decision /opt/sambaflow/apps/nlp/pytorch/transformers_on_rdu/human_decisions/mac_overrides/bertlarge_sc_training_mlm_ml_perf_fullfeature_macv2.json --mac-v2 --non_split_head --dense_adam --data-parallel -ws 2 --max_grad_norm_clip 1.0 --adam_beta2 0.98 --weight_decay 0.01 --pef transformers_hook --output-folder=${OUTDIR}"
echo "COMPILE COMMAND: $COMMAND"
eval $COMMAND

echo "RUN"
COMMAND="/opt/mpich-3.3.2/bin/mpirun -np 8 python /opt/sambaflow/apps/nlp/pytorch/transformers_on_rdu/transformers_hook.py run --config_name /opt/sambaflow/apps/nlp/pytorch/transformers_on_rdu/modules/configs/mlm_24layer_ml_perf_config.json --tokenizer_name bert-large-uncased --module_name mlm_ns --task_name mlm_ns --max_seq_length 128 -b 256 --pef ${OUTDIR}/transformers_hook/transformers_hook.pef --output_dir=${OUTDIR}/hf_output --overwrite_output_dir --do_train --per_device_train_batch_size 256 --input_dir /usr/local/share/data/bert/WIKI_EN_SUBSAMPLE_DIR --cache ${OUTDIR}/cache --max_predictions_per_seq 20 --save_steps -1 --warmup_steps 6250 --max_steps 125000 --steps_this_run 5005 --logging_steps 100 --weight_decay 0.01 --learning_rate 0.00035  --non_split_head --dense_adam --data-parallel --reduce-on-rdu --adam_beta2 0.98 --max_grad_norm_clip 1.0 --validate_stat_perf --validate_tying_plus_embed_train"
echo "RUN COMMAND: $COMMAND"
eval $COMMAND

echo "PERF"
#COMMAND="/opt/mpich-3.3.2/bin/mpirun -hosts sm-02 -np 8 python /opt/sambaflow/apps/nlp/pytorch/transformers_on_rdu/transformers_hook.py measure-performance --config_name /opt/sambaflow/apps/nlp/pytorch/transformers_on_rdu/modules/configs/mlm_24layer_ml_perf_config.json --tokenizer_name bert-large-uncased --module_name mlm_ns --task_name mlm_ns --max_seq_length 128 -b 256 --pef ${OUTDIR}/transformers_hook/transformers_hook.pef --output_dir=${OUTDIR}/hf_output --overwrite_output_dir --do_train --per_device_train_batch_size 256 --input_dir /usr/local/share/data/bert/WIKI_EN_SUBSAMPLE_DIR --cache ${OUTDIR}/cache --max_predictions_per_seq 20 --save_steps -1 --warmup_steps 6250 --max_steps 125000 --steps_this_run 5005 --logging_steps 100 --weight_decay 0.01 --learning_rate 0.00035  --non_split_head --dense_adam --data-parallel --reduce-on-rdu --adam_beta2 0.98 --max_grad_norm_clip 1.0 --validate_stat_perf --validate_tying_plus_embed_train"
COMMAND="/opt/mpich-3.3.2/bin/mpirun -hosts sm-02 -np 2 python /opt/sambaflow/apps/nlp/pytorch/transformers_on_rdu/transformers_hook.py measure-performance --config_name /opt/sambaflow/apps/nlp/pytorch/transformers_on_rdu/modules/configs/mlm_24layer_ml_perf_config.json --tokenizer_name bert-large-uncased --module_name mlm_ns --task_name mlm_ns --max_seq_length 128 -b 256 --pef ${OUTDIR}/transformers_hook/transformers_hook.pef --output_dir=${OUTDIR}/hf_output --overwrite_output_dir --do_train --per_device_train_batch_size 256 --input_dir /usr/local/share/data/bert/WIKI_EN_SUBSAMPLE_DIR --cache ${OUTDIR}/cache --max_predictions_per_seq 20 --save_steps -1 --warmup_steps 6250 --max_steps 125000 --steps_this_run 5005 --logging_steps 100 --weight_decay 0.01 --learning_rate 0.00035  --non_split_head --dense_adam --data-parallel --reduce-on-rdu --adam_beta2 0.98 --max_grad_norm_clip 1.0 --validate_stat_perf --validate_tying_plus_embed_train"
echo "PERF COMMAND: $COMMAND"
eval $COMMAND

echo "Machine state after: "
/opt/sambaflow/bin/snfadm -l inventory

echo "Duration: " $SECONDS
