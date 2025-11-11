#!/bin/bash

MODELS=(
    "claude-3-7-sonnet-20250219-thinking"
    "o4-mini-2025-04-16"
    "gpt-4.1-2025-04-14"
    "gemini-2.0-flash"
    "gemini-2.5-pro-preview-03-25"
    "gpt-4o"
)

REASON_SETTINGS=(
    "with_reason"
    "no_reason"
)

MODES=(
    "global"
    "local"
    "both"
)

LOG_DIR="./batch_logs_0507_0224"
mkdir -p $LOG_DIR

START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
START_SECONDS=$(date +%s)
echo "Batch evaluation started at: $START_TIME"
echo "====================================="

SUMMARY_FILE="$LOG_DIR/batch_summary_$START_TIME.txt"
echo "Batch evaluation summary - $START_TIME" > $SUMMARY_FILE
echo "Model list: ${MODELS[*]}" >> $SUMMARY_FILE
echo "Evaluation mode: ${MODES[*]}" >> $SUMMARY_FILE
echo "====================================" >> $SUMMARY_FILE

for MODEL in "${MODELS[@]}"; do
    for REASON_SETTING in "${REASON_SETTINGS[@]}"; do
        for MODE in "${MODES[@]}"; do
            echo "Start evaluating model: $MODEL - $REASON_SETTING - mode: $MODE"
            
            if [ "$REASON_SETTING" == "with_reason" ]; then
                REASON_FLAG="-r"
                echo "Reason: Yes"
            else
                REASON_FLAG="-n"
                echo "Reason: No"
            fi
            
            LOG_FILE="$LOG_DIR/${MODEL}_${REASON_SETTING}_${MODE}_$(date +"%Y%m%d_%H%M%S").log"
            
            echo "Command: run_score_evaluation.sh -m $MODEL $REASON_FLAG -e $MODE"
            echo "Log file: $LOG_FILE"
            
            EVAL_START_TIME=$(date +"%s")
            bash run_score_evaluation.sh -m "$MODEL" $REASON_FLAG -e "$MODE" > "$LOG_FILE" 2>&1
            EVAL_STATUS=$?
            EVAL_END_TIME=$(date +"%s")
            EVAL_DURATION=$((EVAL_END_TIME - EVAL_START_TIME))
            
            if [ $EVAL_STATUS -eq 0 ]; then
                echo "Evaluation completed, time: $EVAL_DURATION seconds"
                echo "$MODEL - $REASON_SETTING - $MODE: Success (time: $EVAL_DURATION seconds)" >> $SUMMARY_FILE
                
                OUTPUT_DIR=$(grep "Output will be saved to:" "$LOG_FILE" | awk '{print $NF}')
                if [ -n "$OUTPUT_DIR" ]; then
                    echo "  Output directory: $OUTPUT_DIR" >> $SUMMARY_FILE
                    
                    if [ "$MODE" == "global" ]; then
                        SUMMARY_JSON="$OUTPUT_DIR/global/summary.json"
                        if [ -f "$SUMMARY_JSON" ]; then
                            ACCURACY=$(grep -A 1 "Overall accuracy:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
                            if [ -n "$ACCURACY" ]; then
                                echo "  Global evaluation overall accuracy: $ACCURACY" >> $SUMMARY_FILE
                            fi
                            
                            PEARSON=$(grep -A 2 "Overall accuracy:" "$LOG_FILE" | grep "Average Pearson coefficient:" | awk '{print $NF}')
                            if [ -n "$PEARSON" ]; then
                                echo "  Global evaluation average Pearson coefficient: $PEARSON" >> $SUMMARY_FILE
                            fi
                            
                            echo "  Global evaluation dimension details:" >> $SUMMARY_FILE
                            DIMENSIONS=$(grep "Pearson coefficient" "$LOG_FILE" | awk -F':' '{print $1}')
                            for DIM in $DIMENSIONS; do
                                DIM_RESULT=$(grep "$DIM:" "$LOG_FILE")
                                echo "    $DIM_RESULT" >> $SUMMARY_FILE
                            done
                        else
                            echo "  Global evaluation results not found: $SUMMARY_JSON" >> $SUMMARY_FILE
                        fi
                    elif [ "$MODE" == "local" ]; then
                        SUMMARY_JSON="$OUTPUT_DIR/local/summary.json"
                        if [ -f "$SUMMARY_JSON" ]; then
                            ACCURACY=$(grep -A 1 "Overall accuracy:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
                            if [ -n "$ACCURACY" ]; then
                                echo "  Local evaluation overall accuracy: $ACCURACY" >> $SUMMARY_FILE
                            fi
                            
                            PEARSON=$(grep -A 2 "Overall accuracy:" "$LOG_FILE" | grep "Average Pearson coefficient:" | awk '{print $NF}')
                            if [ -n "$PEARSON" ]; then
                                echo "  Local evaluation average Pearson coefficient: $PEARSON" >> $SUMMARY_FILE
                            fi
                            
                            echo "  Local evaluation dimension details:" >> $SUMMARY_FILE
                            DIMENSIONS=$(grep "Pearson coefficient" "$LOG_FILE" | awk -F':' '{print $1}')
                            for DIM in $DIMENSIONS; do
                                DIM_RESULT=$(grep "$DIM:" "$LOG_FILE")
                                echo "    $DIM_RESULT" >> $SUMMARY_FILE
                            done
                        else
                            echo "  Local evaluation results not found: $SUMMARY_JSON" >> $SUMMARY_FILE
                        fi
                    elif [ "$MODE" == "both" ]; then
                        SUMMARY_JSON="$OUTPUT_DIR/combined_summary.json"
                        if [ -f "$SUMMARY_JSON" ]; then
                            GLOBAL_SECTION=$(grep -A 5 "==== Comprehensive Evaluation Results ====" "$LOG_FILE")
                            GLOBAL_ACCURACY=$(echo "$GLOBAL_SECTION" | grep "Global evaluation overall accuracy:" | awk '{print $NF}')
                            GLOBAL_PEARSON=$(echo "$GLOBAL_SECTION" | grep "Global evaluation average Pearson coefficient:" | awk '{print $NF}')
                            
                            if [ -n "$GLOBAL_ACCURACY" ]; then
                                echo "  Global evaluation overall accuracy: $GLOBAL_ACCURACY" >> $SUMMARY_FILE
                            fi
                            if [ -n "$GLOBAL_PEARSON" ]; then
                                echo "  Global evaluation average Pearson coefficient: $GLOBAL_PEARSON" >> $SUMMARY_FILE
                            fi
                            
                        
                            LOCAL_ACCURACY=$(echo "$GLOBAL_SECTION" | grep "Local evaluation overall accuracy:" | awk '{print $NF}')
                            LOCAL_PEARSON=$(echo "$GLOBAL_SECTION" | grep "Local evaluation average Pearson coefficient:" | awk '{print $NF}')
                            
                            if [ -n "$LOCAL_ACCURACY" ]; then
                                echo "  Local evaluation overall accuracy: $LOCAL_ACCURACY" >> $SUMMARY_FILE
                            fi
                            if [ -n "$LOCAL_PEARSON" ]; then
                                echo "  Local evaluation average Pearson coefficient: $LOCAL_PEARSON" >> $SUMMARY_FILE
                            fi
                            
                            GLOBAL_RESULTS_SECTION=$(grep -A 100 "Each dimension's consistency ratio and Pearson correlation coefficient:" "$LOG_FILE" | grep -B 100 "Overall accuracy:" | head -n -1)
                            echo "  Global evaluation dimension details:" >> $SUMMARY_FILE
                            echo "$GLOBAL_RESULTS_SECTION" | grep ":" | while read -r line; do
                                echo "    $line" >> $SUMMARY_FILE
                            done
                            
                            LOCAL_RESULTS_SECTION=$(grep -A 100 "Local evaluation completed!" "$LOG_FILE" | grep -A 100 "Each dimension's consistency ratio and Pearson correlation coefficient:" | grep -B 100 "Overall accuracy:" | head -n -1)
                            if [ -n "$LOCAL_RESULTS_SECTION" ]; then
                                echo "  Local evaluation dimension details:" >> $SUMMARY_FILE
                                echo "$LOCAL_RESULTS_SECTION" | grep ":" | while read -r line; do
                                    echo "    $line" >> $SUMMARY_FILE
                                done
                            fi
                        else
                            echo "  Comprehensive evaluation results not found: $SUMMARY_JSON" >> $SUMMARY_FILE
                        fi
                    fi
                else
                    echo "  Output directory not found" >> $SUMMARY_FILE
                fi
            else
                echo "Evaluation failed, exit code: $EVAL_STATUS"
                echo "$MODEL - $REASON_SETTING - $MODE: Failed (exit code: $EVAL_STATUS)" >> $SUMMARY_FILE
            fi
            
            echo "====================================="
            echo "" >> $SUMMARY_FILE
        done
    done
done

END_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
END_SECONDS=$(date +%s)
TOTAL_DURATION=$((END_SECONDS - START_SECONDS))
echo "Batch evaluation ended at: $END_TIME"
echo "Total duration: $TOTAL_DURATION seconds"
echo "Summary file: $SUMMARY_FILE"

echo "====================================" >> $SUMMARY_FILE
echo "Batch evaluation ended at: $END_TIME" >> $SUMMARY_FILE
echo "Total duration: $TOTAL_DURATION seconds" >> $SUMMARY_FILE

echo -e "\nModel performance comparison table:" >> $SUMMARY_FILE
echo "Model | Reason setting | Evaluation mode | Evaluation type | Accuracy | Pearson coefficient" >> $SUMMARY_FILE
echo "-----|----------|----------|----------|--------|------------" >> $SUMMARY_FILE

for MODEL in "${MODELS[@]}"; do
    for REASON_SETTING in "${REASON_SETTINGS[@]}"; do
        for MODE in "${MODES[@]}"; do
            LOG_FILES=($LOG_DIR/${MODEL}_${REASON_SETTING}_${MODE}_*.log)
            if [ ${#LOG_FILES[@]} -gt 0 ] && [ -f "${LOG_FILES[-1]}" ]; then
                LOG_FILE="${LOG_FILES[-1]}"
                
                if [ "$MODE" == "global" ]; then
                    ACCURACY=$(grep -A 1 "Overall accuracy:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
                    PEARSON=$(grep -A 2 "Overall accuracy:" "$LOG_FILE" | grep "Average Pearson coefficient:" | awk '{print $NF}')
                    
                    if [ -n "$ACCURACY" ]; then
                        echo "$MODEL | $REASON_SETTING | $MODE | Global | $ACCURACY | ${PEARSON:-N/A}" >> $SUMMARY_FILE
                    else
                        echo "$MODEL | $REASON_SETTING | $MODE | Global | Not found | Not found" >> $SUMMARY_FILE
                    fi
                elif [ "$MODE" == "local" ]; then
                    ACCURACY=$(grep -A 1 "Overall accuracy:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
                    PEARSON=$(grep -A 2 "Overall accuracy:" "$LOG_FILE" | grep "Average Pearson coefficient:" | awk '{print $NF}')
                    
                    if [ -n "$ACCURACY" ]; then
                        echo "$MODEL | $REASON_SETTING | $MODE | Local | $ACCURACY | ${PEARSON:-N/A}" >> $SUMMARY_FILE
                    else
                        echo "$MODEL | $REASON_SETTING | $MODE | Local | Not found | Not found" >> $SUMMARY_FILE
                    fi
                elif [ "$MODE" == "both" ]; then
                    GLOBAL_SECTION=$(grep -A 5 "==== Comprehensive Evaluation Results ====" "$LOG_FILE")
                    GLOBAL_ACCURACY=$(echo "$GLOBAL_SECTION" | grep "Global evaluation overall accuracy:" | awk '{print $NF}')
                    GLOBAL_PEARSON=$(echo "$GLOBAL_SECTION" | grep "Global evaluation average Pearson coefficient:" | awk '{print $NF}')
                    
                    if [ -n "$GLOBAL_ACCURACY" ]; then
                        echo "$MODEL | $REASON_SETTING | $MODE | Global | $GLOBAL_ACCURACY | ${GLOBAL_PEARSON:-N/A}" >> $SUMMARY_FILE
                    else
                        echo "$MODEL | $REASON_SETTING | $MODE | Global | Not found | Not found" >> $SUMMARY_FILE
                    fi
                    
                    LOCAL_ACCURACY=$(echo "$GLOBAL_SECTION" | grep "Local evaluation overall accuracy:" | awk '{print $NF}')
                    LOCAL_PEARSON=$(echo "$GLOBAL_SECTION" | grep "Local evaluation average Pearson coefficient:" | awk '{print $NF}')
                    
                    if [ -n "$LOCAL_ACCURACY" ]; then
                        echo "$MODEL | $REASON_SETTING | $MODE | Local | $LOCAL_ACCURACY | ${LOCAL_PEARSON:-N/A}" >> $SUMMARY_FILE
                    else
                        echo "$MODEL | $REASON_SETTING | $MODE | Local | Not found | Not found" >> $SUMMARY_FILE
                    fi
                fi
            else
                echo "$MODEL | $REASON_SETTING | $MODE | All | Not found | Not found" >> $SUMMARY_FILE
            fi
        done
    done
done

echo -e "\nDetailed evaluation results are saved in the output directory of each model." >> $SUMMARY_FILE

echo "Batch processing script execution completed"
