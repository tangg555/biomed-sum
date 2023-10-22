#!/bin/bash
set -e

# =============================== train ====================
# ============ led
#python tasks/summarisation/train.py --model_name sumled --experiment_name=sumled-biomed\
# --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=allenai/led-base-16384 \
# --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
# --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset
#
#python tasks/summarisation/train.py --model_name sumled --experiment_name=pubmedled-biomed\
# --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=Blaise-g/led_pubmed_sumpubmed_1 \
# --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
# --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

python tasks/summarisation/train.py --model_name sumref_led --experiment_name=sumref_pubmedled-biomed\
 --learning_rate=1e-4 --train_batch_size=2 --eval_batch_size=8 --model_name_or_path=Blaise-g/led_pubmed_sumpubmed_1  \
 --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=16  --num_sanity_val_steps=1 \
 --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

python tasks/summarisation/train.py --model_name oneref_led --experiment_name=oneref_led-biomed\
 --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=Blaise-g/led_pubmed_sumpubmed_1 \
 --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
 --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

# ========= pegasus
#python tasks/summarisation/train.py --model_name sumpegasus --experiment_name=sumpegasus-biomed\
# --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=google/pegasus-x-base \
# --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
# --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

#python tasks/summarisation/train.py --model_name sumpegasus --experiment_name=pubmedpegasus-biomed\
# --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=google/pegasus-pubmed \
# --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
# --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

python tasks/summarisation/train.py --model_name sumref_pegasus --experiment_name=sumref_pubmedpegasus-biomed\
 --learning_rate=1e-4 --train_batch_size=2 --eval_batch_size=8 --model_name_or_path=google/pegasus-pubmed \
 --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=16  --num_sanity_val_steps=1 \
 --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset
#
#python tasks/summarisation/train.py --model_name oneref_pegasus --experiment_name=oneref_pubmedpegasus-biomed\
# --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=google/pegasus-pubmed \
# --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
# --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

# ========= bart
#python tasks/summarisation/train.py --model_name sumbart --experiment_name=sumbart-biomed\
# --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=facebook/bart-base \
# --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
# --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

python tasks/summarisation/train.py --model_name sumbart --experiment_name=pubmedbart-biomed\
 --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=mse30/bart-base-finetuned-pubmed \
 --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
 --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

python tasks/summarisation/train.py --model_name sumbart --experiment_name=pubmedbart-biomed\
 --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=resources/bart-pubmed \
 --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
 --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset


#python tasks/summarisation/train.py --model_name sumref_bart --experiment_name=sumref_pubmedbart-biomed\
# --learning_rate=1e-4 --train_batch_size=2 --eval_batch_size=8 --model_name_or_path=mse30/bart-base-finetuned-pubmed \
# --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=4  --num_sanity_val_steps=1 \
# --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

#python tasks/summarisation/train.py --model_name oneref_bart --experiment_name=oneref_pubmedbart-biomed\
# --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=mse30/bart-base-finetuned-pubmed \
# --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
# --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

# =============================== test ====================
# ===== led
#python tasks/summarisation/test.py\
#  --eval_batch_size=16 --model_name_or_path=output/summarisation/sumled-biomed/best_tfmr \
#  --output_dir=output/summarisation --model_name sumled --experiment_name=sumled-biomed --eval_beams 4
#
#python tasks/summarisation/test.py\
#  --eval_batch_size=16 --model_name_or_path=output/summarisation/pubmedled-biomed/best_tfmr \
#  --output_dir=output/summarisation --model_name sumled --experiment_name=pubmedled-biomed --eval_beams 4

python tasks/summarisation/test.py\
  --eval_batch_size=16 --model_name_or_path=output/summarisation/sumref_pubmedled-biomed/best_tfmr \
  --output_dir=output/summarisation --model_name sumref_led --experiment_name=sumref_pubmedled-biomed --eval_beams 4

python tasks/summarisation/test.py\
  --eval_batch_size=16 --model_name_or_path=output/summarisation/oneref_pubmedled-biomed/best_tfmr \
  --output_dir=output/summarisation --model_name oneref_led --experiment_name=oneref_pubmedled-biomed --eval_beams 4


# ===== pegasus
#python tasks/summarisation/test.py\
#  --eval_batch_size=16 --model_name_or_path=output/summarisation/sumpegasus-biomed/best_tfmr \
#  --output_dir=output/summarisation --model_name sumpegasus --experiment_name=sumpegasus-biomed --eval_beams 4

python tasks/summarisation/test.py\
  --eval_batch_size=16 --model_name_or_path=output/summarisation/pubmedpegasus-biomed/best_tfmr \
  --output_dir=output/summarisation --model_name sumpegasus --experiment_name=pubmedpegasus-biomed --eval_beams 4

python tasks/summarisation/test.py\
  --eval_batch_size=16 --model_name_or_path=output/summarisation/sumref_pubmedpegasus-biomed/best_tfmr \
  --output_dir=output/summarisation --model_name sumref_pegasus --experiment_name=sumref_pubmedpegasus-biomed --eval_beams 4

python tasks/summarisation/test.py\
  --eval_batch_size=8 --model_name_or_path=output/summarisation/oneref_pubmedpegasus-biomed/best_tfmr \
  --output_dir=output/summarisation --model_name oneref_pegasus--experiment_name=oneref_pubmedpegasus-biomed --eval_beams 4

# ===== bart
#python tasks/summarisation/test.py\
#  --eval_batch_size=16 --model_name_or_path=output/summarisation/sumbart-biomed/best_tfmr \
#  --output_dir=output/summarisation --model_name sumbart --experiment_name=sumbart-biomed --eval_beams 4

#python tasks/summarisation/test.py\
#  --eval_batch_size=16 --model_name_or_path=output/summarisation/pubmedbart-biomed/best_tfmr \
#  --output_dir=output/summarisation --model_name sumbart --experiment_name=pubmedbart-biomed --eval_beams 4
#
#python tasks/summarisation/test.py\
#  --eval_batch_size=16 --model_name_or_path=output/summarisation/sumref_pubmedbart-biomed/best_tfmr \
#  --output_dir=output/summarisation --model_name sumref_bart --experiment_name=sumref_pubmedbart-biomed --eval_beams 4
#
#python tasks/summarisation/test.py\
#  --eval_batch_size=32 --model_name_or_path=output/summarisation/oneref_pubmedbart-biomed/best_tfmr \
#  --output_dir=output/summarisation --model_name oneref_bart --experiment_name=oneref_pubmedbart-biomed --eval_beams 4




## ========================================= abandoned ==========================================
python tasks/summarisation/train.py --model_name cg_bart --experiment_name=cg_bart-biomed\
 --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=mse30/bart-base-finetuned-pubmed \
 --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
 --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset

python tasks/summarisation/train.py --model_name cg_bart --experiment_name=cg_bart-biomed\
 --learning_rate=1e-4 --train_batch_size=4 --eval_batch_size=16 --model_name_or_path=resources/bart-pubmed \
 --val_check_interval=0.5 --limit_val_batches=10 --max_epochs=10 --accum_batches_args=1  --num_sanity_val_steps=1 \
 --save_top_k 1 --eval_beams 4 --num_beam_groups 0 --data_dir=datasets/summarisation/biomed_ref_dataset



python tasks/summarisation/test.py\
  --eval_batch_size=16 --model_name_or_path=output/summarisation/cg_bart-biomed/best_tfmr \
  --output_dir=output/summarisation --model_name cg_bart --experiment_name=cg_bart-biomed --eval_beams 4
