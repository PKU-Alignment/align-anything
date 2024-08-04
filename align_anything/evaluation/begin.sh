
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python __main__.py \
    -b BBH \
    --output_dir /aifs4su/yaodong/donghai/align-anything/align_anything/evaluation/meta_test_output/bbh \
    -g vllm