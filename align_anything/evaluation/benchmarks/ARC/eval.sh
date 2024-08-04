SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# 手动解析长选项
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)
      output="$2"
      shift 2
      ;;
    --generation_backend)
      backend="$2"
      shift 2
      ;;
    -g)
      backend="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done


if [ "$backend" = "vllm" ]; then
  python vllm_eval.py \
    --output_dir "$output"
else
  deepspeed \
    --module ds_eval \
    --output_dir $output
fi