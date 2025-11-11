#!/bin/bash

# Define the list of models to evaluate
MODELS=(
    "claude-3-7-sonnet-20250219-thinking"
    "o4-mini-2025-04-16"
    "gpt-4.1-2025-04-14"
    "gemini-2.0-flash"
    "gemini-2.5-pro-preview-03-25"
    "gpt-4o"
)

# Define reason settings
REASON_SETTINGS=(
    "with_reason"
    "no_reason"
)

# Define evaluation modes
MODES=(
    "local"
)

# Set maximum number of parallel jobs
MAX_PARALLEL_JOBS=4

# Log directory
LOG_DIR="./batch_pair_logs_$(date +"%m%d_%H%M")"
mkdir -p $LOG_DIR

# Record start time
START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
START_SECONDS=$(date +%s)
echo "Batch evaluation started at: $START_TIME"
echo "====================================="

# Create summary file
SUMMARY_FILE="$LOG_DIR/batch_summary_$START_TIME.txt"
echo "Batch Evaluation Summary - $START_TIME" > $SUMMARY_FILE
echo "Model list: ${MODELS[*]}" >> $SUMMARY_FILE
echo "Evaluation modes: ${MODES[*]}" >> $SUMMARY_FILE
echo "Max parallel jobs: $MAX_PARALLEL_JOBS" >> $SUMMARY_FILE
echo "====================================" >> $SUMMARY_FILE

# Create a temporary file to store process IDs
PIDS_FILE=$(mktemp)

# Create a function to run a single evaluation task
run_evaluation() {
    local MODEL=$1
    local REASON_SETTING=$2
    local MODE=$3
    
    echo "Starting evaluation for model: $MODEL - $REASON_SETTING - Mode: $MODE"
    
    # Set reason parameter
    if [ "$REASON_SETTING" == "with_reason" ]; then
        REASON_FLAG="-r"
        echo "Include reason: Yes"
    else
        REASON_FLAG="-n"
        echo "Include reason: No"
    fi
    
    # Define log file
    LOG_FILE="$LOG_DIR/${MODEL}_${REASON_SETTING}_${MODE}_$(date +"%Y%m%d_%H%M%S").log"
    
    # Execute evaluation and log output
    echo "Executing command: run_pair_evaluation.sh -m $MODEL $REASON_FLAG -e $MODE -l 0"
    echo "Log file: $LOG_FILE"
    
    # Execute evaluation and record start time
    EVAL_START_TIME=$(date +"%s")
    
    # Run evaluation in a subprocess
    (
        bash run_pair_evaluation.sh -m "$MODEL" $REASON_FLAG -e "$MODE" -l 0 > "$LOG_FILE" 2>&1
        EVAL_STATUS=$?
        EVAL_END_TIME=$(date +"%s")
        EVAL_DURATION=$((EVAL_END_TIME - EVAL_START_TIME))
        
        # Write results to a temporary file
        RESULT_FILE=$(mktemp)
        echo "Model: $MODEL - $REASON_SETTING - $MODE" > $RESULT_FILE
        
        if [ $EVAL_STATUS -eq 0 ]; then
            echo "Evaluation completed in: $EVAL_DURATION seconds" >> $RESULT_FILE
            echo "$MODEL - $REASON_SETTING - $MODE: Success (Duration: $EVAL_DURATION seconds)" >> $RESULT_FILE
            
            # Extract output directory
            OUTPUT_DIR=$(grep "Output directory:" "$LOG_FILE" | awk '{print $NF}')
            if [ -n "$OUTPUT_DIR" ]; then
                echo "  Output directory: $OUTPUT_DIR" >> $RESULT_FILE
                
                # Process evaluation results for different modes
                if [ "$MODE" == "global" ]; then
                    # Global evaluation mode
                    SUMMARY_JSON="$OUTPUT_DIR/global/summary.json"
                    if [ -f "$SUMMARY_JSON" ]; then
                        # Extract overall accuracy
                        ACCURACY=$(grep -A 1 "Overall accuracy:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
                        if [ -n "$ACCURACY" ]; then
                            echo "  Global evaluation overall accuracy: $ACCURACY" >> $RESULT_FILE
                        fi
                        
                        # Extract results for each dimension
                        echo "  Global evaluation dimension details:" >> $RESULT_FILE
                        DIMENSIONS=$(grep -o "[a-z_]\+: [0-9.]\+" "$LOG_FILE")
                        echo "$DIMENSIONS" >> $RESULT_FILE
                    else
                        echo "  Could not find global evaluation results: $SUMMARY_JSON" >> $RESULT_FILE
                    fi
                elif [ "$MODE" == "local" ]; then
                    # Local evaluation mode
                    SUMMARY_JSON="$OUTPUT_DIR/local/summary.json"
                    if [ -f "$SUMMARY_JSON" ]; then
                        # Extract overall accuracy
                        ACCURACY=$(grep -A 1 "Overall accuracy:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
                        if [ -n "$ACCURACY" ]; then
                            echo "  Local evaluation overall accuracy: $ACCURACY" >> $RESULT_FILE
                        fi
                        
                        # Extract results for each dimension
                        echo "  Local evaluation dimension details:" >> $RESULT_FILE
                        DIMENSIONS=$(grep -o "[a-z_]\+: [0-9.]\+" "$LOG_FILE")
                        echo "$DIMENSIONS" >> $RESULT_FILE
                    else
                        echo "  Could not find local evaluation results: $SUMMARY_JSON" >> $RESULT_FILE
                    fi
                elif [ "$MODE" == "both" ]; then
                    # Combined evaluation mode
                    SUMMARY_JSON="$OUTPUT_DIR/combined_summary.json"
                    if [ -f "$SUMMARY_JSON" ]; then
                        # Extract global evaluation results
                        GLOBAL_SECTION=$(grep -A 10 "Global evaluation completed!" "$LOG_FILE")
                        GLOBAL_ACCURACY=$(echo "$GLOBAL_SECTION" | grep "Overall accuracy:" | awk '{print $NF}')
                        
                        if [ -n "$GLOBAL_ACCURACY" ]; then
                            echo "  Global evaluation overall accuracy: $GLOBAL_ACCURACY" >> $RESULT_FILE
                        fi
                        
                        # Extract local evaluation results
                        LOCAL_SECTION=$(grep -A 10 "Local evaluation completed!" "$LOG_FILE")
                        LOCAL_ACCURACY=$(echo "$LOCAL_SECTION" | grep "Overall accuracy:" | awk '{print $NF}')
                        
                        if [ -n "$LOCAL_ACCURACY" ]; then
                            echo "  Local evaluation overall accuracy: $LOCAL_ACCURACY" >> $RESULT_FILE
                        fi
                        
                        # Extract global dimension results
                        echo "  Global evaluation dimension details:" >> $RESULT_FILE
                        GLOBAL_DIMENSIONS=$(echo "$GLOBAL_SECTION" | grep -o "[a-z_]\+: [0-9.]\+")
                        echo "$GLOBAL_DIMENSIONS" >> $RESULT_FILE
                        
                        # Extract local dimension results
                        echo "  Local evaluation dimension details:" >> $RESULT_FILE
                        LOCAL_DIMENSIONS=$(echo "$LOCAL_SECTION" | grep -o "[a-z_]\+: [0-9.]\+")
                        echo "$LOCAL_DIMENSIONS" >> $RESULT_FILE
                    else
                        echo "  Could not find combined evaluation results: $SUMMARY_JSON" >> $RESULT_FILE
                    fi
                fi
            else
                echo "  Could not find output directory" >> $RESULT_FILE
            fi
        else
            echo "Evaluation failed, exit code: $EVAL_STATUS" >> $RESULT_FILE
            echo "$MODEL - $REASON_SETTING - $MODE: Failed (Exit code: $EVAL_STATUS)" >> $RESULT_FILE
        fi
        
        # Use a lock file to ensure summary file writes do not conflict
        (
            flock -x 200
            cat $RESULT_FILE >> $SUMMARY_FILE
            echo "" >> $SUMMARY_FILE
            echo "====================================" >> $SUMMARY_FILE
        ) 200>$SUMMARY_FILE.lock
        
        rm $RESULT_FILE
        
        # Output completion message
        echo "Task completed: $MODEL - $REASON_SETTING - $MODE (Duration: $EVAL_DURATION seconds)"
    ) &
    
    # Save process ID
    echo $! >> $PIDS_FILE
    
    # Check and control the number of parallel tasks
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
        # Wait for any task to complete
        sleep 2
    done
}

# Iterate over all models, reason settings, and evaluation modes
TOTAL_JOBS=$((${#MODELS[@]} * ${#REASON_SETTINGS[@]} * ${#MODES[@]}))
CURRENT_JOB=0

echo "Total number of jobs: $TOTAL_JOBS"
echo "Total number of jobs: $TOTAL_JOBS" >> $SUMMARY_FILE

for MODEL in "${MODELS[@]}"; do
    for REASON_SETTING in "${REASON_SETTINGS[@]}"; do
        for MODE in "${MODES[@]}"; do
            CURRENT_JOB=$((CURRENT_JOB + 1))
            echo "Starting job $CURRENT_JOB/$TOTAL_JOBS"
            run_evaluation "$MODEL" "$REASON_SETTING" "$MODE"
        done
    done
done

# Wait for all tasks to complete
echo "Waiting for all evaluation tasks to complete..."
for PID in $(cat $PIDS_FILE); do
    wait $PID
done

# Clean up temporary files
rm $PIDS_FILE
rm -f $SUMMARY_FILE.lock

# Record end time
END_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
END_SECONDS=$(date +%s)
TOTAL_DURATION=$((END_SECONDS - START_SECONDS))
echo "Batch evaluation finished at: $END_TIME"
echo "Total duration: $TOTAL_DURATION seconds"
echo "Summary file: $SUMMARY_FILE"

echo "====================================" >> $SUMMARY_FILE
echo "Batch evaluation finished at: $END_TIME" >> $SUMMARY_FILE
echo "Total duration: $TOTAL_DURATION seconds" >> $SUMMARY_FILE

# Print model performance comparison table
echo -e "
Model Performance Comparison Table:" >> $SUMMARY_FILE
echo "Model | Reason Setting | Eval Mode | Eval Type | Accuracy" >> $SUMMARY_FILE
echo "-----|----------|----------|----------|--------" >> $SUMMARY_FILE

for MODEL in "${MODELS[@]}"; do
    for REASON_SETTING in "${REASON_SETTINGS[@]}"; do
        for MODE in "${MODES[@]}"; do
            LOG_FILES=($LOG_DIR/${MODEL}_${REASON_SETTING}_${MODE}_*.log)
            if [ ${#LOG_FILES[@]} -gt 0 ] && [ -f "${LOG_FILES[-1]}" ]; then
                LOG_FILE="${LOG_FILES[-1]}"
                
                if [ "$MODE" == "global" ]; then
                    # Global mode
                    ACCURACY=$(grep -A 1 "Overall accuracy:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
                    
                    if [ -n "$ACCURACY" ]; then
                        echo "$MODEL | $REASON_SETTING | $MODE | Global Eval | $ACCURACY" >> $SUMMARY_FILE
                    else
                        echo "$MODEL | $REASON_SETTING | $MODE | Global Eval | Not Found" >> $SUMMARY_FILE
                    fi
                elif [ "$MODE" == "local" ]; then
                    # Local mode
                    ACCURACY=$(grep -A 1 "Overall accuracy:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
                    
                    if [ -n "$ACCURACY" ]; then
                        echo "$MODEL | $REASON_SETTING | $MODE | Local Eval | $ACCURACY" >> $SUMMARY_FILE
                    else
                        echo "$MODEL | $REASON_SETTING | $MODE | Local Eval | Not Found" >> $SUMMARY_FILE
                    fi
                elif [ "$MODE" == "both" ]; then
                    # Combined mode - Global results
                    GLOBAL_SECTION=$(grep -A 10 "Global evaluation completed!" "$LOG_FILE")
                    GLOBAL_ACCURACY=$(echo "$GLOBAL_SECTION" | grep "Overall accuracy:" | awk '{print $NF}')
                    
                    if [ -n "$GLOBAL_ACCURACY" ]; then
                        echo "$MODEL | $REASON_SETTING | $MODE | Global Eval | $GLOBAL_ACCURACY" >> $SUMMARY_FILE
                    else
                        echo "$MODEL | $REASON_SETTING | $MODE | Global Eval | Not Found" >> $SUMMARY_FILE
                    fi
                    
                    # Combined mode - Local results
                    LOCAL_SECTION=$(grep -A 10 "Local evaluation completed!" "$LOG_FILE")
                    LOCAL_ACCURACY=$(echo "$LOCAL_SECTION" | grep "Overall accuracy:" | awk '{print $NF}')
                    
                    if [ -n "$LOCAL_ACCURACY" ]; then
                        echo "$MODEL | $REASON_SETTING | $MODE | Local Eval | $LOCAL_ACCURACY" >> $SUMMARY_FILE
                    else
                        echo "$MODEL | $REASON_SETTING | $MODE | Local Eval | Not Found" >> $SUMMARY_FILE
                    fi
                fi
            else
                echo "$MODEL | $REASON_SETTING | $MODE | All | Log Not Found" >> $SUMMARY_FILE
            fi
        done
    done
done

echo -e "
Detailed evaluation results are saved in the output directories of each model." >> $SUMMARY_FILE

echo "Batch script execution completed" 