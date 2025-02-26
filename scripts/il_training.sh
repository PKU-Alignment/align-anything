export OBJAVERSE_HOUSES_BASE_DIR=/home/yuhao/safevla/data/houses/objaverse_houses,
export OBJAVERSE_HOUSES_DIR=/home/yuhao/safevla/data/houses/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_BASE_DIR=/home/yuhao/safevla/data/assets/objaverse_houses
export OBJAVERSE_DATA_DIR=/home/yuhao/safevla/data/assets/objaverse_assets/2023_07_28
export OBJAVERSE_ANNOTATIONS_PATH=/home/yuhao/safevla/data/assets/objaverse_assets/2023_07_28/annotations.json.gz
export WANDB_DIR=/home/yuhao/align-anything/wandb
export HF_ENDPOINT=https://hf-mirror.com

export LONG_ACTION_NAME=1
python -m align_anything.trainers.text_video_to_action.training.offline.train_pl \
 --max_samples 10000000 \
 --eval_max_samples 100 \
 --eval_every 300 \
 --model_version small_3\
 --sliding_window 100 \
 --per_gpu_batch 8 \
 --lr 0.0002 \
 --data_dir /home/yuhao/safevla/data/data/ \
 --dataset_version CHORES \
 --model EarlyFusionCnnTransformer \
 --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
 --precision 16-mixed \
 --resume_local \
 --output_dir il_ckpt \
 --loss action \
 --max_epochs 400