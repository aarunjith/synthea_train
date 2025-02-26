#!/bin/bash

# Script to run the Synthea BERT pipeline with the virtual environment

# Exit on error
set -e

# Set up colored output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Synthea BERT Pipeline ===${NC}"
echo -e "${YELLOW}Using virtual environment in .env${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .env/bin/activate

# Set essential environment variables for tqdm
export PYTHONUNBUFFERED=1        # Unbuffered output
export PYTHONIOENCODING=utf-8    # Proper Unicode handling 
export FORCE_TQDM=1              # Force enable tqdm
export TQDM_MININTERVAL=0.1      # Update interval
export TQDM_SMOOTHING=0.3        # Smoother updates

# Check if we're in an interactive terminal
if [ -t 1 ]; then
  # We're in a terminal - optimize for interactive display
  export TQDM_DISABLE=0                  # Ensure tqdm is enabled
  export TQDM_NESTED=1                   # Allow nested bars
  
  # Use simpler format for terminal
  export TQDM_BAR_FORMAT='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
  
  # Tell Python to use the real terminal size
  stty size > /dev/null 2>&1 && export COLUMNS=$(stty size | cut -d' ' -f2) || export COLUMNS=80
  
  # Check if terminal supports colors
  if [ -n "$TERM" ] && [ "$TERM" != "dumb" ]; then
    export TQDM_COLOUR=AUTO      # Use color if supported
  else
    export TQDM_COLOUR=NO       # No color for dumb terminals
  fi
else
  # We're not in a terminal - use ASCII bars and no animation
  export TQDM_ASCII=1
  export TQDM_DISABLE_LOCK=1
  export TQDM_COLOUR=NO
fi

# Check if wandb is installed, if not install it
if ! python -c "import wandb" &> /dev/null; then
  echo -e "${YELLOW}Installing Weights & Biases (wandb)...${NC}"
  pip install wandb
fi

# Check PyTorch version for torch.compile support
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
TORCH_COMPILE_SUPPORTED=false
if [[ $(python -c "import torch; from packaging import version; print(version.parse(torch.__version__) >= version.parse('2.0.0'))") == "True" ]]; then
  TORCH_COMPILE_SUPPORTED=true
  echo -e "${GREEN}PyTorch $TORCH_VERSION supports torch.compile${NC}"
else
  echo -e "${YELLOW}PyTorch $TORCH_VERSION does not support torch.compile (requires 2.0+)${NC}"
fi

# Default parameters
INPUT_DIR="fhir/fhir"
OUTPUT_DIR="./runs"
BATCH_SIZE=16
NUM_EPOCHS=20
LEARNING_RATE=0.00005
NUM_WORKERS=4
SKIP_PROCESSED=""
DISABLE_WANDB=""
DISABLE_INFERENCE=""
COMPILE=""
COMPILE_MODE="auto"
EXISTING_TSV=""
MAX_STEPS_PER_EPOCH=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --skip_processed)
      SKIP_PROCESSED="--skip_processed"
      shift
      ;;
    --disable_wandb)
      DISABLE_WANDB="--disable_wandb"
      shift
      ;;
    --disable_inference)
      DISABLE_INFERENCE="--disable_inference"
      shift
      ;;
    --existing_tsv)
      EXISTING_TSV="--existing_tsv $2"
      shift 2
      ;;
    --compile)
      if $TORCH_COMPILE_SUPPORTED; then
        COMPILE="--compile"
      else
        echo -e "${YELLOW}Ignoring --compile flag as PyTorch $TORCH_VERSION does not support it${NC}"
      fi
      shift
      ;;
    --compile_mode)
      COMPILE_MODE="$2"
      shift 2
      ;;
    --max_steps_per_epoch)
      MAX_STEPS_PER_EPOCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Run the pipeline
echo -e "${GREEN}Running pipeline with:${NC}"
echo -e "  Input directory: ${YELLOW}$INPUT_DIR${NC}"
echo -e "  Output directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "  Batch size: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  Num epochs: ${YELLOW}$NUM_EPOCHS${NC}"
echo -e "  Learning rate: ${YELLOW}$LEARNING_RATE${NC}"
echo -e "  Num workers: ${YELLOW}$NUM_WORKERS${NC}"
if [[ -n "$EXISTING_TSV" ]]; then
  # Extract just the file path from the flag
  TSV_PATH=$(echo "$EXISTING_TSV" | cut -d ' ' -f 2)
  echo -e "  Using existing TSV: ${YELLOW}$TSV_PATH${NC}"
fi
if [[ -n "$SKIP_PROCESSED" ]]; then
  echo -e "  Skip processed: ${YELLOW}Yes${NC}"
fi
if [[ -n "$DISABLE_WANDB" ]]; then
  echo -e "  W&B logging: ${YELLOW}Disabled${NC}"
else
  echo -e "  W&B logging: ${YELLOW}Enabled${NC}"
fi
if [[ -n "$DISABLE_INFERENCE" ]]; then
  echo -e "  Inference testing: ${YELLOW}Disabled${NC}"
else
  echo -e "  Inference testing: ${YELLOW}Enabled${NC}"
fi
if [[ -n "$COMPILE" ]]; then
  echo -e "  torch.compile: ${YELLOW}Enabled${NC}"
  echo -e "  Compilation mode: ${YELLOW}$COMPILE_MODE${NC}"
else
  echo -e "  torch.compile: ${YELLOW}Disabled${NC}"
fi
if [[ $MAX_STEPS_PER_EPOCH -gt 0 ]]; then
  echo -e "  Max steps per epoch: ${YELLOW}$MAX_STEPS_PER_EPOCH${NC}"
else
  echo -e "  Max steps per epoch: ${YELLOW}No limit${NC}"
fi

echo -e "${GREEN}Starting pipeline...${NC}"

# Ensure there are extra lines so progress bars don't overwrite status
echo ""
echo ""

# Execute directly instead of using a subshell to keep terminal control
# This ensures tqdm can properly update the terminal
exec python run_pipeline.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --num_workers "$NUM_WORKERS" \
  $SKIP_PROCESSED \
  $DISABLE_WANDB \
  $DISABLE_INFERENCE \
  $COMPILE \
  ${COMPILE:+--compile_mode "$COMPILE_MODE"} \
  $EXISTING_TSV \
  ${MAX_STEPS_PER_EPOCH:+--max_steps_per_epoch "$MAX_STEPS_PER_EPOCH"}

echo -e "${GREEN}Pipeline completed!${NC}" 