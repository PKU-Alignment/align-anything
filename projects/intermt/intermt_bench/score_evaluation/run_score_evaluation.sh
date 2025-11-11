#!/bin/bash

# Default parameter values
MODEL="gemini-2.5-pro-preview-03-25"
TEMPERATURE=0.2
# Default categories for global and local evaluation
GLOBAL_CATEGORIES="context_awareness,helpfulness,crucial_step_recognition,global_image_text_consistency,style_coherence"
LOCAL_CATEGORIES="local_image_text_consistency,visual_perceptual_quality,text_quality,context_coherence"
# Default to global evaluation categories
CATEGORIES=$GLOBAL_CATEGORIES
REASON=true
TEST_FILE=""
OUTPUT_DIR="./output_0507_0224"
CACHE_DIR="./cache"
LIMIT=0
MODE="both"

# Show usage instructions
function show_usage {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model MODEL         Model name to use (default: $MODEL)"
    echo "  -t, --temperature TEMP    Generation temperature (default: $TEMPERATURE)"
    echo "  -c, --categories CATS     Evaluation categories, comma-separated, or use 'all' (default: auto-selected based on evaluation mode)"
    echo "  -r, --reason              Include reasoning (default)"
    echo "  -n, --no-reason           Do not include reasoning"
    echo "  -f, --test-file FILE      Test dataset path (default: $TEST_FILE)"
    echo "  -o, --output-dir DIR      Output directory (default: $OUTPUT_DIR)"
    echo "  -d, --cache-dir DIR       Cache directory (default: $CACHE_DIR)"
    echo "  -l, --limit NUM           Limit number of samples to process, 0 means process all samples (default: $LIMIT)"
    echo "  -e, --mode MODE           Evaluation mode: global(overall evaluation), local(per-turn evaluation), both(both modes) (default: $MODE)"
    echo "  -h, --help                Show this help message"
    exit 1
}

# Function to set default categories based on mode
function set_categories_by_mode {
    local mode=$1
    local user_categories=$2
    
    # If user hasn't explicitly specified categories, set default categories based on mode
    if [ -z "$user_categories" ]; then
        case $mode in
            global)
                CATEGORIES=$GLOBAL_CATEGORIES
                ;;
            local)
                CATEGORIES=$LOCAL_CATEGORIES
                ;;
            both)
                CATEGORIES="all" # Use "all", let inference.py handle category selection
                ;;
            *)
                echo "Warning: Unknown mode '$mode', using global evaluation categories"
                CATEGORIES=$GLOBAL_CATEGORIES
                ;;
        esac
    else
        # User specified categories, use user's choice
        CATEGORIES=$user_categories
    fi
}

# Initialize user-specified categories variable
USER_CATEGORIES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -c|--categories)
            USER_CATEGORIES="$2"
            shift 2
            ;;
        -r|--reason)
            REASON=true
            shift
            ;;
        -n|--no-reason)
            REASON=false
            shift
            ;;
        -f|--test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -e|--mode)
            MODE="$2"
            # Set default categories based on mode (if user hasn't specified)
            set_categories_by_mode "$2" "$USER_CATEGORIES"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# If user specified categories, use user-specified categories
if [ -n "$USER_CATEGORIES" ]; then
    CATEGORIES=$USER_CATEGORIES
else
    # If user hasn't specified categories, ensure default categories are set based on mode
    set_categories_by_mode "$MODE" "$USER_CATEGORIES"
fi

# Build argument string
ARGS="--model $MODEL --temperature $TEMPERATURE --categories $CATEGORIES --mode $MODE"

if $REASON; then
    ARGS="$ARGS --reason"
else
    ARGS="$ARGS --no-reason"
fi

ARGS="$ARGS --test-file $TEST_FILE --output-dir $OUTPUT_DIR --cache-dir $CACHE_DIR"

if [ $LIMIT -gt 0 ]; then
    ARGS="$ARGS --limit $LIMIT"
fi

# Print parameter information
echo "Running evaluation with the following parameters:"
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Evaluation mode: $MODE"
echo "Evaluation categories: $CATEGORIES"
echo "Include reasoning: $REASON"
echo "Test file: $TEST_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Cache directory: $CACHE_DIR"
if [ $LIMIT -gt 0 ]; then
    echo "Sample limit: $LIMIT"
else
    echo "Sample limit: No limit"
fi

# Execute inference evaluation
echo "Executing: python score_inference.py $ARGS"
python score_inference.py $ARGS

echo "Evaluation completed!" 