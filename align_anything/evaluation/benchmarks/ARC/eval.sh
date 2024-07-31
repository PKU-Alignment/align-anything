export CUDA_HOME="/aifs4su/yaodong/miniconda3/envs/xuyao-dev-eval/"

# python ds_eval.py \
#   --output_dir /aifs4su/yaodong/xuyao/evaluation/align-anything-eval/align_anything/evaluation/meta_test_output

deepspeed \
  --module ds_eval \
  --output_dir /aifs4su/yaodong/xuyao/evaluation/align-anything-eval/align_anything/evaluation/meta_test_output \