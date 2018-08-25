from . import my_universal_transformer


# num trainable parameters: 59982848

# CUDA_VISIBLE_DEVICES="0" t2t-trainer  \
# --t2t_usr_dir=~/Universal_Transformers/t2t_usr_dir \
# --data_dir=~/t2t_data/wmt_ende/data  \
# --tmp_dir=~/t2t_data/wmt_ende/tmp   \
# --output_dir=~/t2t_data/wmt_ende/output_ut_gru  \
# --problem=translate_ende_wmt32k   \
# --model=my_universal_transformer   \
# --hparams_set=universal_transformer_with_gru_as_transition_function   \
# --save_checkpoints_secs=1800   \
# --keep_checkpoint_max=50  \
#  --train_steps=9000000   \
#  --eval_steps=20 \
# --worker_gpu=1


# num trainable parameters: 64179200

# CUDA_VISIBLE_DEVICES="1" t2t-trainer  \
# --t2t_usr_dir=~/Universal_Transformers/t2t_usr_dir \
# --data_dir=~/t2t_data/wmt_ende/data  \
# --tmp_dir=~/t2t_data/wmt_ende/tmp   \
# --output_dir=~/t2t_data/wmt_ende/output_ut_lstm  \
# --problem=translate_ende_wmt32k   \
# --model=my_universal_transformer   \
# --hparams_set=universal_transformer_with_lstm_as_transition_function   \
# --save_checkpoints_secs=1800   \
# --keep_checkpoint_max=50  \
#  --train_steps=9000000   \
#  --eval_steps=20 \
# --worker_gpu=1


# num trainable parameters: 76821504

# CUDA_VISIBLE_DEVICES="2" t2t-trainer  \
# --t2t_usr_dir=~/Universal_Transformers/t2t_usr_dir \
# --data_dir=~/t2t_data/wmt_ende/data  \
# --tmp_dir=~/t2t_data/wmt_ende/tmp   \
# --output_dir=~/t2t_data/wmt_ende/output_ut_basic_plus_gru  \
# --problem=translate_ende_wmt32k   \
# --model=my_universal_transformer   \
# --hparams_set=universal_transformer_basic_plus_gru   \
# --save_checkpoints_secs=1800   \
# --keep_checkpoint_max=50  \
#  --train_steps=9000000   \
#  --eval_steps=20 \
# --worker_gpu=1



# num trainable parameters: 81017856

# CUDA_VISIBLE_DEVICES="3" t2t-trainer  \
# --t2t_usr_dir=~/Universal_Transformers/t2t_usr_dir \
# --data_dir=~/t2t_data/wmt_ende/data  \
# --tmp_dir=~/t2t_data/wmt_ende/tmp   \
# --output_dir=~/t2t_data/wmt_ende/output_ut_basic_plus_lstm  \
# --problem=translate_ende_wmt32k   \
# --model=my_universal_transformer   \
# --hparams_set=universal_transformer_basic_plus_lstm   \
# --save_checkpoints_secs=1800   \
# --keep_checkpoint_max=50  \
#  --train_steps=9000000   \
#  --eval_steps=20 \
# --worker_gpu=1


# num trainable parameters: 66335744

# CUDA_VISIBLE_DEVICES="2,3" t2t-trainer  \
# --t2t_usr_dir=~/Universal_Transformers/t2t_usr_dir \
# --data_dir=~/t2t_data/wmt_ende/data  \
# --tmp_dir=~/t2t_data/wmt_ende/tmp   \
# --output_dir=~/t2t_data/wmt_ende/output_ut_all_steps_so_far \
# --problem=translate_ende_wmt32k   \
# --model=my_universal_transformer   \
# --hparams_set=universal_transformer_all_steps_so_far   \
# --save_checkpoints_secs=1800   \
# --keep_checkpoint_max=50  \
#  --train_steps=9000000   \
#  --eval_steps=20 \
# --worker_gpu=2