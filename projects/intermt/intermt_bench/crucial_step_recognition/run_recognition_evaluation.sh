#!/bin/bash

# Set base directory
BASE_DIR="./"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define color codes
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
MAGENTA="\033[0;35m"
CYAN="\033[0;36m"
BOLD="\033[1m"
UNDERLINE="\033[4m"
RESET="\033[0m"

# Create output and cache directories
OUTPUT_ROOT="${BASE_DIR}/batch_recognition_logs"
OUTPUT_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"
CACHE_DIR="${OUTPUT_ROOT}/cache_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Log recording
LOG_FILE="${OUTPUT_DIR}/run_log.txt"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "\n${BOLD}${BLUE}=========================================================${RESET}"
echo -e "${BOLD}${BLUE}           Crucial Step Recognition Batch Evaluation      ${RESET}"
echo -e "${BOLD}${BLUE}=========================================================${RESET}"
echo -e "${BOLD}Start time:${RESET} $(date)"
echo -e "${BOLD}Output directory:${RESET} ${CYAN}$OUTPUT_DIR${RESET}"
echo -e "${BOLD}Cache directory:${RESET} ${CYAN}$CACHE_DIR${RESET}"

# Define inference model list
declare -a INFERENCE_MODELS=(
    "claude-3-7-sonnet-20250219-thinking"
    "o4-mini-2025-04-16"
    "gpt-4.1-2025-04-14"
    # "gemini-2.0-flash"
    # "gemini-2.5-pro-preview-03-25"
    # "gpt-4o"
)

# Fixed judge model
JUDGE_MODEL="gpt-4o"

# Fixed temperature
TEMPERATURE=0.7

# Sample count limit (0 means use all samples)
SAMPLE_LIMIT=100

# Display evaluation configuration
echo -e "\n${BOLD}${YELLOW}Evaluation Configuration:${RESET}"
echo -e "${BOLD}Judge Model:${RESET} ${MAGENTA}$JUDGE_MODEL${RESET}"
echo -e "${BOLD}Temperature:${RESET} ${MAGENTA}$TEMPERATURE${RESET}"
echo -e "${BOLD}Sample Limit:${RESET} ${MAGENTA}$SAMPLE_LIMIT${RESET}"

# Create model result directories
echo -e "\n${BOLD}${YELLOW}Creating model result directories...${RESET}"
for model in "${INFERENCE_MODELS[@]}"; do
    mkdir -p "${OUTPUT_DIR}/${model}"
    echo -e "- ${CYAN}${model}${RESET} directory created"
done

# Start timing
START_TIME=$(date +%s)

# Run all model evaluations
total_models=${#INFERENCE_MODELS[@]}
current_model=1

for model in "${INFERENCE_MODELS[@]}"; do
    model_start_time=$(date +%s)
    
    echo -e "\n${BOLD}${GREEN}=========================================================${RESET}"
    echo -e "${BOLD}${GREEN}Running Model [$current_model/$total_models]: ${CYAN}$model${RESET}${RESET}"
    echo -e "${BOLD}${GREEN}=========================================================${RESET}"
    
    # Run evaluation script
    python "${BASE_DIR}/recognition_inference.py" \
        --inference-model "$model" \
        --judge-model "$JUDGE_MODEL" \
        --temperature "$TEMPERATURE" \
        --output-dir "${OUTPUT_DIR}/${model}" \
        --cache-dir "$CACHE_DIR" \
        --limit "$SAMPLE_LIMIT"
    
    # Check run status
    model_end_time=$(date +%s)
    model_elapsed=$((model_end_time - model_start_time))
    model_minutes=$((model_elapsed / 60))
    model_seconds=$((model_elapsed % 60))
    
    if [ $? -eq 0 ]; then
        echo -e "\n${BOLD}${GREEN}✓ Successfully completed $model evaluation${RESET} - Time: ${model_minutes}m${model_seconds}s"
    else
        echo -e "\n${BOLD}${RED}✗ $model evaluation failed${RESET} - Time: ${model_minutes}m${model_seconds}s"
    fi
    
    # Show progress
    percent_done=$(( (current_model * 100) / total_models ))
    bar_done=$((percent_done / 2))
    bar_left=$((50 - bar_done))
    
    # Create progress bar
    progress_bar=""
    for ((i=0; i<bar_done; i++)); do
        progress_bar="${progress_bar}█"
    done
    for ((i=0; i<bar_left; i++)); do
        progress_bar="${progress_bar}░"
    done
    
    echo -e "${BOLD}${BLUE}[$progress_bar] $percent_done%${RESET}"
    
    # If not the last model, show estimated remaining time
    if [ $current_model -lt $total_models ]; then
        avg_time_per_model=$((model_elapsed))
        remaining_models=$((total_models - current_model))
        est_remaining_time=$((avg_time_per_model * remaining_models))
        est_remaining_minutes=$((est_remaining_time / 60))
        est_remaining_seconds=$((est_remaining_time % 60))
        
        echo -e "${BOLD}Estimated remaining time:${RESET} approximately ${est_remaining_minutes}m${est_remaining_seconds}s"
    fi
    
    current_model=$((current_model + 1))
done

# Calculate total elapsed time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(( (TOTAL_TIME % 3600) / 60 ))
SECONDS=$((TOTAL_TIME % 60))

echo -e "\n${BOLD}${GREEN}=========================================================${RESET}"
echo -e "${BOLD}${GREEN}               All Model Evaluations Complete           ${RESET}"
echo -e "${BOLD}${GREEN}=========================================================${RESET}"
echo -e "${BOLD}Start time:${RESET} $(date -d @$START_TIME)"
echo -e "${BOLD}End time:${RESET} $(date)"
echo -e "${BOLD}Total time:${RESET} ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo -e "${BOLD}Results saved in:${RESET} ${CYAN}$OUTPUT_DIR${RESET}"

# Generate run summary
SUMMARY_FILE="${OUTPUT_DIR}/evaluation_summary.txt"
echo -e "${BOLD}${BLUE}Batch Evaluation Run Summary${RESET}" > "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"
echo "Run time: $(date)" >> "$SUMMARY_FILE"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$SUMMARY_FILE"
echo -e "\nConfiguration:" >> "$SUMMARY_FILE"
echo "- Judge model: $JUDGE_MODEL" >> "$SUMMARY_FILE"
echo "- Temperature: $TEMPERATURE" >> "$SUMMARY_FILE"
echo "- Sample limit: $SAMPLE_LIMIT" >> "$SUMMARY_FILE"
echo -e "\nEvaluated models:" >> "$SUMMARY_FILE"
for model in "${INFERENCE_MODELS[@]}"; do
    echo "- $model" >> "$SUMMARY_FILE"
done

# Collect evaluation results from all models
echo -e "\nEvaluation Results Summary:" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"

# Create model result comparison table
echo -e "\n${BOLD}${BLUE}Model Evaluation Results Comparison:${RESET}"
echo -e "${BOLD}-------------------------------------------------------------${RESET}"
echo -e "${BOLD}| No.  | Model Name                       | Avg Score | Samples |${RESET}"
echo -e "${BOLD}-------------------------------------------------------------${RESET}"

model_index=1
best_score=0
best_model=""

for model in "${INFERENCE_MODELS[@]}"; do
    stats_file="${OUTPUT_DIR}/${model}/statistics.json"
    if [ -f "$stats_file" ]; then
        echo -e "\nModel: $model" >> "$SUMMARY_FILE"
        avg_score=$(jq '.average_score' "$stats_file")
        total_eval=$(jq '.total_evaluated' "$stats_file")
        total_samples=$(jq '.total_samples' "$stats_file")
        
        # Get score distribution
        score_dist=$(jq -r '.score_distribution | to_entries | .[] | "\(.key): \(.value)"' "$stats_file")
        
        echo "- Average score: $avg_score" >> "$SUMMARY_FILE"
        echo "- Evaluated samples: $total_eval / $total_samples" >> "$SUMMARY_FILE"
        echo "- Score distribution:" >> "$SUMMARY_FILE"
        echo "$score_dist" | while read -r line; do
            echo "  $line" >> "$SUMMARY_FILE"
        done
        
        # Output to console table
        model_display=$(printf "%-30s" "$model")
        printf "| %4d | ${CYAN}%s${RESET} | ${MAGENTA}%6.2f${RESET} | %6d |\n" "$model_index" "$model_display" "$avg_score" "$total_eval"
        
        # Update best model
        if (( $(echo "$avg_score > $best_score" | bc -l) )); then
            best_score=$avg_score
            best_model=$model
        fi
    else
        echo -e "\nModel: $model" >> "$SUMMARY_FILE"
        echo "- Evaluation failed or incomplete" >> "$SUMMARY_FILE"
        
        # Output to console table
        model_display=$(printf "%-30s" "$model")
        printf "| %4d | ${CYAN}%s${RESET} | ${RED}Failed${RESET}  |    -   |\n" "$model_index" "$model_display"
    fi
    
    model_index=$((model_index + 1))
done

echo -e "${BOLD}-------------------------------------------------------------${RESET}"

# Show best model
if [ -n "$best_model" ]; then
    echo -e "\n${BOLD}${GREEN}Best performing model:${RESET} ${CYAN}$best_model${RESET} - Average score: ${MAGENTA}$best_score${RESET}"
    echo -e "\nBest performing model: $best_model (score: $best_score)" >> "$SUMMARY_FILE"
fi

# Create symlink for latest results
LATEST_LINK="${OUTPUT_ROOT}/latest"
rm -f "$LATEST_LINK"
ln -s "$OUTPUT_DIR" "$LATEST_LINK"

echo -e "\n${BOLD}Evaluation summary saved to:${RESET} ${CYAN}$SUMMARY_FILE${RESET}"
echo -e "${BOLD}Latest results accessible via:${RESET} ${CYAN}$LATEST_LINK${RESET}"

# Add sound notification to indicate evaluation completion
echo -e "\a"
echo -e "\n${BOLD}${GREEN}All evaluation tasks completed!${RESET}" 