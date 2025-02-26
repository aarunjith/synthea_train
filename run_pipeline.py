#!/usr/bin/env python3
import os
import argparse
import subprocess
import logging
import glob
import sys
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Path to Python in virtual environment
PYTHON_EXECUTABLE = os.path.join(os.getcwd(), ".env", "bin", "python")
if not os.path.exists(PYTHON_EXECUTABLE):
    logger.warning(f"Virtual environment Python at {PYTHON_EXECUTABLE} not found, using system Python")
    PYTHON_EXECUTABLE = "python"

def is_interactive():
    """Check if we're in an interactive terminal."""
    return os.isatty(sys.stdout.fileno()) if hasattr(sys.stdout, 'fileno') else False

def run_command(command, description):
    """Run a command and log its output with proper tqdm handling."""
    logger.info(f"{description} - Running: {' '.join(command)}")
    
    # Set environment variables for proper tqdm display
    env = os.environ.copy()
    
    # Critical settings for unbuffered output and proper Unicode
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    
    # Settings for tqdm
    env["FORCE_TQDM"] = "1"              # Force tqdm to show progress bars
    env["TQDM_MININTERVAL"] = "0.1"      # Minimum update interval in seconds
    
    # Detect if we're in an interactive terminal
    is_tty = is_interactive()
    
    if is_tty:
        # For interactive terminals, use in-place updates
        try:
            # Direct execution for interactive terminals to preserve control sequences
            result = subprocess.run(
                command,
                env=env,
                check=True
            )
            logger.info(f"{description} - Completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"{description} - Failed with code {e.returncode}")
            raise Exception(f"{description} failed with code {e.returncode}")
    else:
        # For non-interactive terminals, capture and log output
        try:
            # Run the command with output capturing
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=True
            )
            
            # Log the output
            logger.info(f"{description} - Output:\n{result.stdout}")
            logger.info(f"{description} - Completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"{description} - Failed with code {e.returncode}")
            if e.stdout:
                logger.error(f"Output:\n{e.stdout}")
            raise Exception(f"{description} failed with code {e.returncode}")

def get_device_flags():
    """Check for available GPU and return appropriate device flags."""
    try:
        import torch
        if torch.cuda.is_available():
            return ["--mixed_precision"]
        else:
            return ["--no_cuda"]
    except ImportError:
        return ["--no_cuda"]

def run_pipeline(args):
    """Run the complete training pipeline."""
    output_root = args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Process FHIR files (unless skipped)
    if args.existing_tsv:
        logger.info(f"Skipping FHIR processing, using existing TSV file: {args.existing_tsv}")
        output_tsv = args.existing_tsv
    else:
        logger.info("STEP 1: Processing FHIR files")
        
        output_tsv = os.path.join(output_dir, "combined_claims.tsv")
        
        process_cmd = [
            PYTHON_EXECUTABLE, "batch_process_fhir.py",
            "--input_dir", args.input_dir,
            "--output_file", output_tsv
        ]
        
        if args.skip_processed:
            process_cmd.append("--skip_processed")
        
        run_command(process_cmd, "FHIR processing")
        
        # Check if the TSV file was created and has content
        if not os.path.exists(output_tsv) or os.path.getsize(output_tsv) < 100:
            logger.error(f"The output TSV file {output_tsv} was not created or is empty. Aborting.")
            return
    
    # Step 2: Train the model
    logger.info("STEP 2: Training the model")
    
    train_cmd = [
        PYTHON_EXECUTABLE, "enhanced_train.py",
        "--data_file", output_tsv,
        "--output_dir", os.path.join(output_dir, "model"),
        "--cache_dir", os.path.join(output_dir, "cache"),
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--num_workers", str(args.num_workers)
    ]
    
    # Add device flags
    train_cmd.extend(get_device_flags())
    
    # Add wandb flag if disabled
    if args.disable_wandb:
        train_cmd.append("--disable_wandb")
    
    # Add disable_inference flag if specified
    if args.disable_inference:
        train_cmd.append("--disable_inference")
        
    # Add sampling flags if specified
    if args.sample_percentage < 100:
        train_cmd.extend(["--sample_percentage", str(args.sample_percentage)])
        
    if args.max_rows > 0:
        train_cmd.extend(["--max_rows", str(args.max_rows)])
    
    # Add torch.compile flags
    if args.compile:
        train_cmd.append("--compile")
        if args.compile_mode != "auto":
            train_cmd.extend(["--compile_mode", args.compile_mode])
    
    # Add max_steps_per_epoch flag if specified
    if args.max_steps_per_epoch > 0:
        train_cmd.extend(["--max_steps_per_epoch", str(args.max_steps_per_epoch)])
    
    # Run the training command
    run_command(train_cmd, "Model training")
    
    logger.info(f"Pipeline completed successfully! Output saved to {output_dir}")
    
    # Print the model path for easy reference
    model_path = os.path.join(output_dir, "model", "final_model")
    logger.info(f"Trained model saved to: {model_path}")
    
    # Return the path to the output directory
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete Synthea BERT pipeline")
    
    # Input and output arguments
    parser.add_argument("--input_dir", default="fhir/fhir", help="Directory containing FHIR JSON files")
    parser.add_argument("--output_dir", default="./runs", help="Directory to save output")
    parser.add_argument("--existing_tsv", type=str, help="Path to existing TSV file to skip FHIR processing")
    
    # Processing arguments
    parser.add_argument("--skip_processed", action="store_true", help="Skip already processed files")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Performance arguments
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (requires PyTorch 2.0+)")
    parser.add_argument("--compile_mode", type=str, default="auto", choices=["auto", "reduce-overhead", "max-autotune"], 
                       help="Compilation mode for torch.compile")
    parser.add_argument("--max_steps_per_epoch", type=int, default=0, 
                        help="Maximum steps per epoch (0 = no limit)")
    
    # Wandb arguments
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    
    # Inference arguments
    parser.add_argument("--disable_inference", action="store_true", help="Disable example inference testing")
    
    # Sampling arguments
    parser.add_argument("--sample_percentage", type=float, default=100, help="Percentage of data to sample")
    parser.add_argument("--max_rows", type=int, default=0, help="Maximum number of rows to sample")
    
    args = parser.parse_args()
    
    try:
        output_dir = run_pipeline(args)
        logger.info(f"Pipeline completed successfully! Results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        exit(1) 