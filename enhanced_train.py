import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import numpy as np
import os
import json
import argparse
import logging
import sys
from pathlib import Path
from dataset import ClaimsMaskedDataset, StreamingClaimsMaskedDataset, SequentialStreamingClaimsMaskedDataset
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import wandb
import pkg_resources
import random
import gc
import signal
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure tqdm to work properly in all environments
TQDM_BAR_FORMAT = '{l_bar}{bar:30}{r_bar}'  # Simpler format for better compatibility
# Disable color if it causes issues
os.environ['TQDM_COLOUR'] = 'WHITE'  # Use a specific color instead of AUTO
# Set Python multiprocessing variables to help with cleanup issues
os.environ['PYTHONMALLOC'] = 'debug'  # Help detect memory issues
os.environ['PYTHONASYNCIO'] = 'debug'  # More debug info for async operations
os.environ['PYTHONUTF8'] = '1'  # Avoid encoding issues
os.environ['PYTHONIOENCODING'] = 'utf-8'  # Ensure proper encoding
# Try to make multiprocessing clean up after itself better
os.environ['TORCH_USE_RTLD_GLOBAL'] = '1'  # Help with torch multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads which can cause conflicts

IS_NOTEBOOK = 'ipykernel' in sys.modules
IS_INTERACTIVE = os.isatty(sys.stdout.fileno()) if hasattr(sys.stdout, 'fileno') else False

# Check if torch version supports compilation
TORCH_VERSION = pkg_resources.get_distribution("torch").version
TORCH_COMPILE_SUPPORTED = False
if TORCH_VERSION >= "2.0.0":
    TORCH_COMPILE_SUPPORTED = True
    logger.info(f"PyTorch {TORCH_VERSION} supports compilation with torch.compile")
else:
    logger.warning(f"PyTorch {TORCH_VERSION} does not support torch.compile (requires 2.0+)")

def load_data(data_file, tokenizer, cache_dir="./cache", max_length=128, validation_split=0.1, force_reload=False, sample_percentage=100, max_rows=0):
    """
    Load and preprocess data for the model, using a sequential streaming approach.
    
    Returns:
        train_dataloader, val_dataloader
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"Loading data from {data_file}")
    logger.info(f"Using sequential streaming loader for {data_file}")
    
    # Start timing data processing
    start_time = time.time()
    
    # Check if the data file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Use sequential streaming dataset for much faster processing
    batch_size = args.batch_size  # Default to args.batch_size
    
    # Create training dataset
    train_dataset = SequentialStreamingClaimsMaskedDataset(
        file_path=data_file,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size
    )
    
    # Create validation dataset with a separate file handle
    val_dataset = SequentialStreamingClaimsMaskedDataset(
        file_path=data_file,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size
    )
    
    # Skip ahead in validation dataset to avoid overlap with training
    # (This is a simple way to create a validation set without shuffling)
    val_lines_to_skip = int(train_dataset.line_count * (1 - validation_split))
    for _ in range(val_lines_to_skip):
        val_dataset.file.readline()
    val_dataset.current_line = val_lines_to_skip
    
    logger.info(f"Training dataset: Processing from beginning of file")
    logger.info(f"Validation dataset: Starting from line {val_lines_to_skip:,}")
    
    # Log important statistics about dataset size and examples
    num_columns = train_dataset.estimated_examples_per_line
    total_examples = train_dataset.total_examples
    train_examples = int(total_examples * (1 - validation_split))
    val_examples = int(total_examples * validation_split)
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total lines: {train_dataset.line_count:,}")
    logger.info(f"  Average examples (columns) per line: {num_columns:.2f}")
    logger.info(f"  Estimated total examples: {total_examples:,}")
    logger.info(f"  Training examples: ~{train_examples:,}")
    logger.info(f"  Validation examples: ~{val_examples:,}")
    
    # Worker initialization function to ensure clean process state
    def worker_init_fn(worker_id):
        # Set different random seed for each worker
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        # Make cleanup more aggressive in worker processes
        signal.signal(signal.SIGTERM, signal.SIG_DFL)  # Use default handler for clean exit
        
        # Set process name for easier debugging
        try:
            import setproctitle
            setproctitle.setproctitle(f"worker-{worker_id}")
        except ImportError:
            pass
    
    # Choose appropriate multiprocessing start method for better stability
    mp_start_method = 'fork'
    # If on macOS, use 'spawn' which is more stable there
    if sys.platform == 'darwin':
        mp_start_method = 'spawn'
    
    # Determine if we should use multiprocessing
    use_mp = args.num_workers > 0
    
    # Fix for pytorch sharing in Linux
    if use_mp and sys.platform.startswith('linux'):
        # Ensure shared memory is properly configured
        try:
            torch.multiprocessing.set_sharing_strategy('file_system')
        except RuntimeError:
            logger.warning("Could not set sharing strategy to file_system")
    
    # Create DataLoaders that don't shuffle (sequential processing)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle as we're processing sequentially
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=mp_start_method if use_mp else None,
        persistent_workers=use_mp,  # Keep workers alive between batches
        prefetch_factor=2 if use_mp else None  # Control prefetching
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle as we're processing sequentially
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=mp_start_method if use_mp else None,
        persistent_workers=use_mp,  # Keep workers alive between batches
        prefetch_factor=2 if use_mp else None  # Control prefetching
    )
    
    # Report elapsed time for data processing
    elapsed_time = time.time() - start_time
    logger.info(f"Dataset preparation completed in {elapsed_time:.2f} seconds")
    
    # Log dataloader details
    logger.info(f"Created dataloaders with:")
    logger.info(f"  Training batches: {len(train_dataloader):,}")
    logger.info(f"  Validation batches: {len(val_dataloader):,}") 
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Total training capacity: {len(train_dataloader) * batch_size:,} examples")
    
    return train_dataloader, val_dataloader

def create_dataloaders(dataset, train_sampler, val_sampler, batch_size, num_workers):
    """Create DataLoader objects for training and validation."""
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders with batch_size={batch_size}, num_workers={num_workers}")
    logger.info(f"Training batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")
    logger.info(f"Total training examples: {len(train_sampler):,}")
    
    return train_dataloader, val_dataloader

def validate(model, dataloader, device):
    """Validate the model and return average validation loss."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # Track validation metrics
    losses = []
    
    # Calculate number of validation steps from the actual dataloader length
    # We'll use a percentage of the validation set for faster validation
    val_percentage = 0.1  # Process 10% of the validation set
    val_batches = min(
        max(10, int(len(dataloader) * val_percentage)),  # At least 10 batches but at most 10% of validation set
        1000  # Hard cap at 1000 batches to keep validation time reasonable
    )
    
    logger.info(f"Running validation on {val_batches} batches (out of {len(dataloader):,} total validation batches)")
    
    # Create a separate tqdm progress bar for validation
    val_progress_bar = tqdm(
        range(val_batches),
        desc="Validation", 
        leave=False,  # Don't leave the progress bar when done
        bar_format=TQDM_BAR_FORMAT,
        dynamic_ncols=True,  # Adapt to terminal width
        disable=not IS_INTERACTIVE  # Only show in interactive sessions
    )
    
    try:
        # Create a fresh iterator for validation
        dataloader_iter = iter(dataloader)
        
        validation_start_time = time.time()
        
        with torch.no_grad():
            for step in val_progress_bar:
                # Get next batch from validation dataloader
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    # This shouldn't happen unless we've configured val_batches incorrectly
                    logger.warning(f"Validation dataloader exhausted early at step {step}/{val_batches}")
                    if step == 0:
                        # If we can't even get one batch, return a default loss
                        return 10.0, 0
                    break
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Count non-padding tokens
                num_tokens = torch.sum(attention_mask).item()
                total_tokens += num_tokens
                
                with autocast(enabled=args.mixed_precision):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                loss = outputs.loss
                losses.append(loss.item())
                total_loss += loss.item()
                
                # Calculate current averages
                current_avg_loss = total_loss / (step + 1)
                
                # Update validation progress bar
                val_progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg': f"{current_avg_loss:.4f}",
                    'ppl': f"{np.exp(current_avg_loss):.2f}",
                    'line': f"{dataloader.dataset.current_line}"
                })
                
                # Log validation steps to W&B
                if not args.disable_wandb and step % 5 == 0:  # Only log every 5 steps to avoid too many points
                    wandb.log({
                        'validation_step/loss': loss.item(),
                        'validation_step/avg_loss': current_avg_loss,
                        'validation_step/perplexity': np.exp(loss.item()),
                        'validation_step/tokens': num_tokens
                    })
                
        # Calculate validation metrics
        validation_time = time.time() - validation_start_time
        tokens_per_second = total_tokens / max(0.1, validation_time)
        avg_loss = total_loss / val_batches if val_batches > 0 else 10.0
        
        # Log validation summary
        logger.info(f"Validation completed: Loss={avg_loss:.4f}, Perplexity={np.exp(avg_loss):.2f}, " +
                    f"Tokens/sec={tokens_per_second:.1f}")
        
        # Log validation metrics to W&B
        if not args.disable_wandb:
            wandb.log({
                'validation/avg_loss': avg_loss,
                'validation/perplexity': np.exp(avg_loss),
                'validation/total_tokens': total_tokens,
                'validation/tokens_per_second': tokens_per_second,
                'validation/time_seconds': validation_time
            })
    finally:
        # Ensure proper cleanup
        del dataloader_iter
        # Force garbage collection to clean up resources
        gc.collect()
        
    model.train()
    return avg_loss, total_tokens

def save_checkpoint(model, optimizer, scheduler, epoch, args, val_loss, best=False):
    """Save a training checkpoint."""
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # For compiled models, we need to save the original model state
    if hasattr(model, '_orig_mod'):
        model_state = model._orig_mod.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'args': vars(args)
    }
    
    if best:
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load a checkpoint and return the epoch to start from."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'] + 1, checkpoint.get('val_loss', float('inf'))

def plot_training_progress(train_losses, val_losses, output_dir):
    """Plot the training and validation loss curves."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    logger.info(f"Loss plot saved to {plot_path}")

def sample_masked_token(input, logits, tokenizer, candidates=[]):
    """
    Sample tokens for masked positions and calculate candidate scores.
    
    Args:
        input: Input tensor with token IDs
        logits: Model output logits
        tokenizer: Tokenizer for decoding
        candidates: List of candidate strings to score
        
    Returns:
        If candidates provided: (decoded_tokens, candidate_scores)
            - decoded_tokens: List of decoded tokens for masked positions
            - candidate_scores: List of lists, one list of scores per input sequence
        If no candidates: decoded_tokens
    """
    logits = logits.detach().cpu()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = probs.numpy()
    all_tokens = []
    candidate_scores = []
    
    # For each sequence in the batch
    for i in range(input.shape[0]):
        tokens = []
        mask_positions = torch.where(input[i] == tokenizer.mask_token_id)[0]
        
        for pos in mask_positions:
            token_probs = probs[i, pos]
            predicted_id = np.argmax(token_probs)
            tokens.append(predicted_id)
        
        all_tokens.append(tokens)
        
        # If candidates are provided, compute their scores
        if candidates:
            encoded_candidates = [tokenizer.encode(cand, add_special_tokens=False) for cand in candidates]
            scores = []
            
            for cand_ids in encoded_candidates:
                if len(cand_ids) == len(mask_positions):
                    score = 1.0
                    for j, cid in enumerate(cand_ids):
                        score *= probs[i, mask_positions[j], cid]
                    scores.append(score)
                else:
                    scores.append(0.0)
            
            candidate_scores.append(scores)
    
    # Return appropriate output based on whether candidates were provided
    if candidates:
        return tokenizer.batch_decode(all_tokens), candidate_scores
    return tokenizer.batch_decode(all_tokens)

def run_example_inference(model, tokenizer, test_cases, device):
    """Run inference on example test cases to evaluate model performance."""
    logger.info("Running example inference tests...")
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for case in test_cases:
            query = case["query"].replace("\t", "<tab>").strip("\n")
            candidates = case.get("candidates", [])
            
            # Get model predictions
            inputs = tokenizer(query, return_tensors='pt', padding='max_length', truncation=True).to(device)
            logits = model(**inputs).logits
            
            if candidates:
                predictions, candidate_scores = sample_masked_token(inputs.input_ids, logits, tokenizer, candidates)
                
                # Fix: candidate_scores is a list of lists, need to handle properly
                # If there's only one set of scores (most common case), use the first one
                if len(candidate_scores) == 1:
                    scores_to_log = candidate_scores[0]
                    result = {
                        "query": case["query"],
                        "candidates": candidates,
                        "predictions": predictions,
                        "scores": [float(score) for score in scores_to_log]
                    }
                else:
                    # Handle multiple sets of scores (multiple masked positions)
                    result = {
                        "query": case["query"],
                        "candidates": candidates,
                        "predictions": predictions,
                        "scores": [[float(s) for s in score_set] for score_set in candidate_scores]
                    }
            else:
                predictions = sample_masked_token(inputs.input_ids, logits, tokenizer)
                result = {
                    "query": case["query"],
                    "predictions": predictions
                }
            
            results.append(result)
            
            # Log the results
            logger.info(f"\nQuery: {case['query']}")
            logger.info(f"Predictions: {predictions}")
            if candidates:
                # Fix logging to handle the nested structure
                if len(candidate_scores) == 1:
                    for i, (candidate, score) in enumerate(zip(candidates, candidate_scores[0])):
                        logger.info(f"Candidate {i+1}: {candidate}, Score: {score:.4f}")
                else:
                    logger.info("Multiple mask positions detected, showing candidate scores:")
                    for i, score_set in enumerate(candidate_scores):
                        logger.info(f"  Position {i+1}: {[f'{s:.4f}' for s in score_set]}")
    
    # Save results to a JSON file
    results_path = os.path.join(args.output_dir, "inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Inference results saved to {results_path}")
    model.train()

def get_cosine_with_hard_restarts_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, num_cycles=1, min_lr_ratio=0.0):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: Number of cycles (usually kept at 1).
        min_lr_ratio: Minimum learning rate ratio (final_lr / initial_lr).
        
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        # Safety check - ensure step doesn't exceed total steps
        current_step = min(current_step, num_training_steps)
        
        # Warmup phase
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
            
        # Progress after warmup (ensure this is between 0 and 1)
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, max(0.0, progress))  # Clamp between 0 and 1
        
        # Cosine decay to min_lr instead of 0
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale from 1.0 to min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

def main(args):
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If multiprocessing is disabled, set num_workers to 0
    if args.disable_multiprocessing:
        logger.info("Multiprocessing disabled via command line option")
        args.num_workers = 0
        
    # Initialize Weights & Biases with more complete configuration
    if not args.disable_wandb:
        wandb.init(
            project="synthea_train",
            name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
            # Add tags for easier filtering
            tags=["synthea", f"model-{args.model_name.split('/')[-1]}", f"workers-{args.num_workers}"]
        )
        # Log system info
        wandb.config.update({
            "system": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "gpu_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            }
        })
        
        # Set up custom W&B panels/charts
        wandb.define_metric("epoch")
        # Define train loss as grouped with epoch
        wandb.define_metric("epoch/*", step_metric="epoch")
        # Define validation metrics grouped with epoch
        wandb.define_metric("validation/*", step_metric="epoch")
        # Define training step metrics
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
    
    # Set up TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set up mixed precision scaler
    scaler = GradScaler(enabled=args.mixed_precision)
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {args.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForMaskedLM.from_pretrained(args.model_name)
    
    # Add tab token
    special_tokens = {'additional_special_tokens': ['<tab>']}
    tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added <tab> special token (ID: {tokenizer.convert_tokens_to_ids('<tab>')})")
    model.resize_token_embeddings(len(tokenizer))
    
    # Compile the model if enabled and supported
    if args.compile and TORCH_COMPILE_SUPPORTED:
        logger.info("Compiling model with torch.compile for faster training...")
        try:
            # Select the appropriate mode based on the backend
            if args.compile_mode == "auto":
                compile_mode = "default"  # Let PyTorch choose the best mode
            else:
                compile_mode = args.compile_mode
                
            model = torch.compile(
                model, 
                mode=compile_mode,
                fullgraph=False  # Set to False for HuggingFace models
            )
            logger.info(f"Model successfully compiled with mode: {compile_mode}")
        except Exception as e:
            logger.error(f"Model compilation failed: {str(e)}")
            logger.warning("Continuing with uncompiled model")
    elif args.compile and not TORCH_COMPILE_SUPPORTED:
        logger.warning("Model compilation requested but torch.compile is not supported in this PyTorch version")
        logger.warning("Continuing with uncompiled model")
    
    # Load data
    logger.info("Loading and processing dataset...")
    start_time = time.time()
    train_dataloader, val_dataloader = load_data(
        args.data_file, 
        tokenizer, 
        cache_dir=args.cache_dir,
        max_length=args.max_length,
        validation_split=args.validation_split,
        force_reload=args.force_reload_data,
        sample_percentage=args.sample_percentage,
        max_rows=args.max_rows
    )
    data_load_time = time.time() - start_time
    
    # Report detailed dataset statistics
    logger.info(f"Dataset processing completed in {data_load_time:.2f} seconds")
    logger.info(f"Training batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")
    logger.info(f"Batch size: {args.batch_size}, Total training examples: {len(train_dataloader) * args.batch_size}")
    
    # Set up optimizer with learning rates from command line arguments
    initial_lr = args.initial_lr
    final_lr = args.final_lr
    
    # Log the learning rate settings
    logger.info(f"Setting up cosine learning rate schedule: initial={initial_lr}, final={final_lr}")
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_dataloader) * args.num_epochs
    
    # Validate and adjust warmup ratio if necessary
    if args.warmup_ratio <= 0 or args.warmup_ratio >= 1:
        logger.warning(f"Invalid warmup ratio {args.warmup_ratio}, setting to default 0.1")
        args.warmup_ratio = 0.1
    elif args.warmup_ratio > 0.2:
        logger.warning(f"Warmup ratio {args.warmup_ratio} is quite large. For most training scenarios, " 
                      f"0.05-0.1 is recommended. Setting to 0.1.")
        args.warmup_ratio = 0.1

    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Warn if warmup steps seem excessive
    if warmup_steps > total_steps / 3:
        logger.warning(f"Warmup steps ({warmup_steps}) is more than 1/3 of total steps ({total_steps}). "
                      f"This may cause the learning rate to increase for too long.")
        
    logger.info(f"Learning rate warmup: {warmup_steps}/{total_steps} steps ({args.warmup_ratio:.1%} of training)")
    
    # Calculate the min_lr_ratio (final_lr / initial_lr)
    min_lr_ratio = final_lr / initial_lr
    
    # Use custom scheduler to ensure we reach exactly final_lr
    scheduler = get_cosine_with_hard_restarts_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio  # This ensures we decay to exactly final_lr
    )
    
    # Log the scheduler change
    logger.info(f"Using custom cosine learning rate scheduler with warmup ({warmup_steps} steps)")
    logger.info(f"Learning rate will start at {initial_lr} and decay to exactly {final_lr}")
    
    # Check expected learning rate at key points to verify scheduler is working correctly
    test_steps = [0, warmup_steps // 2, warmup_steps, 
                  warmup_steps + (total_steps - warmup_steps) // 4,
                  warmup_steps + (total_steps - warmup_steps) // 2,
                  total_steps - 1]
                  
    logger.info("Verifying learning rate schedule at key steps:")
    
    # Create a test scheduler for verification
    test_optimizer = AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial_lr)
    test_scheduler = get_cosine_with_hard_restarts_schedule_with_min_lr(
        test_optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio
    )
    
    # Test different steps and log the expected learning rate
    for step in test_steps:
        # Move scheduler to this step
        for _ in range(step):
            test_scheduler.step()
        current_lr = test_scheduler.get_last_lr()[0]
        
        if step < warmup_steps:
            phase = "warmup"
        else:
            phase = "decay"
            
        logger.info(f"  Step {step:5d} ({phase}): lr = {current_lr:.8f}")
        
        # Reset scheduler for next test
        test_optimizer = AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial_lr)
        test_scheduler = get_cosine_with_hard_restarts_schedule_with_min_lr(
            test_optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )
        
    # Visualize the learning rate schedule
    if not args.disable_wandb:
        # Generate the learning rate schedule for visualization
        # Sample fewer points for efficiency (100 points total)
        lr_schedule = []
        sample_points = min(100, total_steps)  # Don't sample more than 100 points
        step_size = max(1, total_steps // sample_points)
        
        # Create a temporary scheduler for visualization
        temp_optimizer = AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial_lr)
        temp_scheduler = get_cosine_with_hard_restarts_schedule_with_min_lr(
            temp_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )
        
        # Sample points from the schedule
        for step in range(0, total_steps, step_size):
            # Fast-forward the scheduler
            for _ in range(step):
                temp_scheduler.step()
            # Get the current learning rate
            current_lr = temp_scheduler.get_last_lr()[0]
            lr_schedule.append([step, current_lr])
            
            # Reset scheduler for next sample
            temp_optimizer = AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial_lr)
            temp_scheduler = get_cosine_with_hard_restarts_schedule_with_min_lr(
                temp_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio
            )
            
        # Ensure we capture key points for accurate visualization
        # Start, end of warmup, middle, and end of training
        key_points = [
            (0, initial_lr * (0.0/max(1, warmup_steps))),  # Start of training (near 0)
            (warmup_steps//2, initial_lr * (0.5)),         # Middle of warmup
            (warmup_steps, initial_lr),                    # End of warmup (peak learning rate)
            (total_steps, final_lr)                        # End of training
        ]
        
        for step, lr in key_points:
            lr_schedule.append([step, lr])
        
        # Sort and deduplicate points
        lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
        lr_schedule = [lr_schedule[i] for i in range(len(lr_schedule)) 
                      if i == 0 or lr_schedule[i][0] != lr_schedule[i-1][0]]
        
        # Log the learning rate schedule to wandb
        lr_table = wandb.Table(data=lr_schedule, columns=["step", "learning_rate"])
        wandb.log({
            "lr_schedule": wandb.plot.line(
                lr_table, "step", "learning_rate", title="Learning Rate Schedule")
        })
    
    # Move model to device
    model = model.to(device)
    
    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    total_tokens_processed = 0
    
    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            start_epoch, best_val_loss = load_checkpoint(
                model, optimizer, scheduler, args.resume_from_checkpoint
            )
            logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint found at {args.resume_from_checkpoint}, starting from scratch")
    
    # Define test cases for periodic evaluation if inference is enabled
    test_cases = []
    if not args.disable_inference:
        test_cases = [
            {
                "query": "active\tpharmacy\tnormal\t1\t<mask><mask><mask><mask><mask><mask><mask><mask>\t0.45\t2015-08-22\tAMB\t\tcomplete\t0.45\tfemale\t1983-05-07\t41\tBlack or African American\tHispanic or Latino\tMarried\tSpanish\tFalse",
                "candidates": ["lisinopril 10 MG Oral Tablet"]
            },
            {
                "query": "active\tpharmacy\tnormal\t1\t<mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask>\t0.45\t2015-08-22\tAMB\t\tcomplete\t0.45\tfemale\t1983-05-07\t41\tBlack or African American\tHispanic or Latino\tMarried\tSpanish\tFalse",
                "candidates": ["Hydrochlorothiazide 25 MG Oral Tablet"]
            }
        ]
        logger.info("Example inference testing is enabled")
    else:
        logger.info("Example inference testing is disabled")
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    no_improvement_count = 0
    
    # Track training time
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.num_epochs):
        total_loss = 0
        epoch_tokens = 0
        epoch_start_time = time.time()
        
        # Reset datasets at the beginning of each epoch
        # (Only needed for the training dataset - validation runs on a separate segment)
        logger.info(f"Resetting training dataset for epoch {epoch + 1}")
        train_dataloader.dataset.reset()
        
        # Set number of batch steps for this epoch using the actual dataloader length
        total_batches = len(train_dataloader)
        
        logger.info(f"Processing {total_batches:,} batches in epoch {epoch+1} with batch size {args.batch_size}")
        
        # Create a tqdm progress bar with enhanced visibility settings
        progress_bar = tqdm(
            range(total_batches),  # Use the actual number of batches
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            leave=True,  # Leave this progress bar when done as it's the main epoch bar
            position=0,  # Ensure it stays at the top position
            bar_format=TQDM_BAR_FORMAT,
            dynamic_ncols=True,  # Adapt to terminal width
            disable=not IS_INTERACTIVE  # Only show in interactive sessions
        )
        
        # Create a single iterator for the epoch
        train_iter = iter(train_dataloader)
        
        # Training step
        for step in progress_bar:
            # Get next batch from the iterator
            try:
                batch = next(train_iter)
            except StopIteration:
                logger.warning(f"Dataloader exhausted at step {step}/{total_batches} (this shouldn't happen)")
                break
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Count non-padding tokens
            batch_tokens = torch.sum(attention_mask).item()
            epoch_tokens += batch_tokens
            
            # Forward pass with mixed precision
            with autocast(enabled=args.mixed_precision):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass with mixed precision
            optimizer.zero_grad()
            
            # Get current learning rate for logging before the step
            current_lr = scheduler.get_last_lr()[0]
            
            if args.mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # Call scheduler after optimizer
                scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                # Call scheduler after optimizer
                scheduler.step()
            
            # Calculate average loss and tokens per second
            avg_loss = total_loss / (step + 1)
            tokens_per_sec = epoch_tokens / max(0.1, time.time() - epoch_start_time)
            
            # Update progress bar with more detailed information
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg': f"{avg_loss:.4f}",
                'lr': f"{current_lr:.2e}",  # Added learning rate display
                'tok/s': f"{tokens_per_sec:.0f}",
                'line': f"{train_dataloader.dataset.current_line:,}"
            })
            
            # Print stats every 100 steps for visibility in log files, but not during interactive sessions
            if step % 100 == 0 and not IS_INTERACTIVE:
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs} - Step {step}/{total_batches} - "
                    f"Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}, Line: {train_dataloader.dataset.current_line:,}"
                )
            
            # Log to TensorBoard
            global_step = epoch * total_batches + step
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('train/line_position', train_dataloader.dataset.current_line, global_step)
            
            # Log to W&B with more comprehensive metrics
            if not args.disable_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/batch_tokens': batch_tokens,
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/line_position': train_dataloader.dataset.current_line,
                    'train/epoch': epoch + (step / total_batches),  # Fractional epoch
                    'train/avg_loss': avg_loss,
                    'train/perplexity': np.exp(loss.item()),
                    'global_step': global_step
                })
            
            # For very large datasets, we might want to break after a certain number of steps
            # to avoid extremely long epochs
            if args.max_steps_per_epoch > 0 and step >= args.max_steps_per_epoch:
                logger.info(f"Reached max steps per epoch ({args.max_steps_per_epoch}), ending epoch early")
                break
        
        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / (step + 1)
        train_losses.append(avg_train_loss)
        
        # Explicitly force garbage collection to clean up any file handles
        gc.collect()
        
        # Update total tokens processed
        total_tokens_processed += epoch_tokens
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        tokens_per_second = epoch_tokens / epoch_time
        
        # Validate after each epoch
        val_loss, val_tokens = validate(model, val_dataloader, device)
        val_losses.append(val_loss)
        
        # Print epoch statistics with clear formatting
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Validation Loss: {val_loss:.4f}")
        logger.info(f"  Perplexity: {np.exp(val_loss):.2f}")
        logger.info(f"  Tokens processed this epoch: {epoch_tokens:,}")
        logger.info(f"  Total tokens processed: {total_tokens_processed:,}")
        logger.info(f"  Epoch time: {epoch_time:.2f}s")
        logger.info(f"  Processing speed: {tokens_per_second:.2f} tokens/second")
        
        # Log to TensorBoard
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('epoch/val_loss', val_loss, epoch)
        writer.add_scalar('epoch/perplexity', np.exp(val_loss), epoch)
        writer.add_scalar('epoch/tokens_processed', epoch_tokens, epoch)
        writer.add_scalar('epoch/total_tokens_processed', total_tokens_processed, epoch)
        writer.add_scalar('epoch/tokens_per_second', tokens_per_second, epoch)
        
        # Log to W&B with more comprehensive epoch metrics
        if not args.disable_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch/train_loss': avg_train_loss,
                'epoch/val_loss': val_loss,
                'epoch/train_perplexity': np.exp(avg_train_loss),
                'epoch/val_perplexity': np.exp(val_loss),
                'epoch/tokens_processed': epoch_tokens,
                'epoch/total_tokens_processed': total_tokens_processed,
                'epoch/tokens_per_second': tokens_per_second,
                'epoch/epoch_time': epoch_time,
                'epoch/learning_rate': scheduler.get_last_lr()[0],
                'epoch/no_improvement_count': no_improvement_count
            })
            
            # Log validation metrics separately to create a custom chart
            wandb.log({
                'validation/loss': val_loss,
                'validation/perplexity': np.exp(val_loss),
                'validation/tokens': val_tokens
            })
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, args, val_loss)
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            logger.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, args, val_loss, best=True)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            logger.info(f"No improvement for {no_improvement_count} epochs")
        
        # Run example inference every few epochs if enabled
        if not args.disable_inference and (epoch + 1) % args.inference_interval == 0:
            run_example_inference(model, tokenizer, test_cases, device)
        
        # Early stopping
        if args.early_stopping > 0 and no_improvement_count >= args.early_stopping:
            logger.info(f"Early stopping triggered after {no_improvement_count} epochs without improvement")
            break
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Total tokens processed: {total_tokens_processed:,}")
    logger.info(f"Average processing speed: {total_tokens_processed / total_training_time:.2f} tokens/second")
    
    # Create final training summary with all key metrics
    final_metrics = {
        'training/total_time': total_training_time,
        'training/total_tokens': total_tokens_processed,
        'training/avg_tokens_per_second': total_tokens_processed / total_training_time,
        'training/epochs_completed': epoch + 1,
        'training/best_val_loss': best_val_loss,
        'training/best_val_perplexity': np.exp(best_val_loss),
        'training/final_train_loss': train_losses[-1] if train_losses else float('nan'),
        'training/final_val_loss': val_losses[-1] if val_losses else float('nan'),
    }
    
    # Log final stats to W&B with more details
    if not args.disable_wandb:
        # Create a custom wandb summary panel
        wandb.run.summary.update(final_metrics)
        
        # Create and log loss curve plot
        if train_losses and val_losses:
            epochs = list(range(1, len(train_losses) + 1))
            loss_plot_data = [[x, y] for (x, y) in zip(epochs, train_losses)]
            val_loss_plot_data = [[x, y] for (x, y) in zip(epochs, val_losses)]
            
            loss_table = wandb.Table(data=loss_plot_data, columns=["epoch", "train_loss"])
            val_loss_table = wandb.Table(data=val_loss_plot_data, columns=["epoch", "val_loss"])
            
            wandb.log({
                "training_curves/loss_plot": wandb.plot.line(
                    loss_table, "epoch", "train_loss", title="Training Loss Curve"),
                "training_curves/val_loss_plot": wandb.plot.line(
                    val_loss_table, "epoch", "val_loss", title="Validation Loss Curve")
            })
            
            # Also log perplexity curves
            train_ppl = [np.exp(loss) for loss in train_losses]
            val_ppl = [np.exp(loss) for loss in val_losses]
            
            train_ppl_data = [[x, y] for (x, y) in zip(epochs, train_ppl)]
            val_ppl_data = [[x, y] for (x, y) in zip(epochs, val_ppl)]
            
            train_ppl_table = wandb.Table(data=train_ppl_data, columns=["epoch", "train_perplexity"])
            val_ppl_table = wandb.Table(data=val_ppl_data, columns=["epoch", "val_perplexity"])
            
            wandb.log({
                "training_curves/train_perplexity_plot": wandb.plot.line(
                    train_ppl_table, "epoch", "train_perplexity", title="Training Perplexity Curve"),
                "training_curves/val_perplexity_plot": wandb.plot.line(
                    val_ppl_table, "epoch", "val_perplexity", title="Validation Perplexity Curve")
            })
        
        # Finish the wandb run
        wandb.finish()
    
    # Plot training progress
    plot_training_progress(train_losses, val_losses, args.output_dir)
    
    # Final evaluation on test cases if enabled
    if not args.disable_inference:
        run_example_inference(model, tokenizer, test_cases, device)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    
    # For compiled models, we need to save the original model state
    if hasattr(model, '_orig_mod'):
        model._orig_mod.save_pretrained(final_model_path)
    else:
        model.save_pretrained(final_model_path)
    
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Final cleanup to prevent temp directory issues
    train_dataset = train_dataloader.dataset
    val_dataset = val_dataloader.dataset
    
    # Explicitly close datasets before exiting
    if hasattr(train_dataset, 'close'):
        train_dataset.close()
    
    if hasattr(val_dataset, 'close'):
        val_dataset.close()
    
    # Force garbage collection
    del train_dataloader
    del val_dataloader
    gc.collect()
    
    # Cleanup temp directories as a safety measure
    from dataset import cleanup_temp_dirs
    cleanup_temp_dirs()
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RoBERTa model for masked language modeling on claims data")
    
    # Data arguments
    parser.add_argument("--data_file", type=str, default="combined_claims.tsv", help="Path to the TSV data file")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to cache processed data")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Proportion of data to use for validation")
    parser.add_argument("--force_reload_data", action="store_true", help="Force reprocessing the data")
    parser.add_argument("--sample_percentage", type=float, default=100, help="Percentage of the dataset to use (1-100)")
    parser.add_argument("--max_rows", type=int, default=0, help="Maximum number of rows to process (0 = no limit)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="distilroberta-base", help="Model name or path")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--initial_lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--final_lr", type=float, default=3e-5, help="Final learning rate (min value for cosine scheduler)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (legacy parameter, use initial_lr instead)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Proportion of training to perform warmup")
    parser.add_argument("--early_stopping", type=int, default=3, help="Stop after this many epochs without improvement")
    parser.add_argument("--inference_interval", type=int, default=5, help="Run inference every N epochs")
    
    # Performance arguments
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (requires PyTorch 2.0+)")
    parser.add_argument("--compile_mode", type=str, default="auto", choices=["auto", "reduce-overhead", "max-autotune"], 
                        help="Compilation mode for torch.compile")
    parser.add_argument("--disable_multiprocessing", action="store_true", 
                        help="Disable multiprocessing completely (sets num_workers=0) to avoid temp directory issues")
    
    # Logging arguments
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--disable_inference", action="store_true", help="Disable example inference testing")
    
    # Add this to the argument parser section
    parser.add_argument("--max_steps_per_epoch", type=int, default=0, 
                        help="Maximum steps per epoch (0 = no limit, process entire dataset)")
    
    args = parser.parse_args()
    
    # Print all arguments
    logger.info("Training arguments:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    main(args) 