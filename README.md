# Synthea BERT - FHIR Claims Masked Language Modeling

This repository contains a pipeline for training RoBERTa models to predict masked content in FHIR claims data. The system can be used to predict missing medication information or other fields in claims data.

## Overview

The pipeline consists of two main parts:

1. **Data Processing**: Converting FHIR JSON files to a structured claims TSV format
2. **Model Training**: Training a masked language model (RoBERTa) to predict missing fields in claims data

## Requirements

- Python 3.7+
- PyTorch 1.7+ (PyTorch 2.0+ recommended for torch.compile support)
- Transformers 4.0+
- pandas
- tqdm
- matplotlib
- tensorboard
- wandb

A Python virtual environment is already set up in the `.env` directory with all required dependencies.

## Quick Start

The easiest way to run the complete pipeline is using the provided shell script:

```bash
# Run with default parameters
./run_synthea_bert.sh

# Or with custom parameters
./run_synthea_bert.sh --input_dir fhir/fhir --output_dir ./runs --batch_size 32 --num_epochs 10

# Enable torch.compile for faster training (requires PyTorch 2.0+)
./run_synthea_bert.sh --compile --compile_mode reduce-overhead
```

Alternatively, you can use the Python script directly:

```bash
# The script automatically uses the virtual environment in .env
python run_pipeline.py --input_dir fhir/fhir --output_dir ./runs

# With torch.compile enabled (requires PyTorch 2.0+)
python run_pipeline.py --input_dir fhir/fhir --output_dir ./runs --compile
```

This will:
1. Process all FHIR JSON files in the `fhir/fhir` directory
2. Train a RoBERTa model on the extracted claims data
3. Save the trained model and results in a timestamped directory under `./runs`

## Step-by-Step Usage

If you prefer to run each step individually:

### 1. Activate the Virtual Environment

```bash
# On Linux/Mac
source .env/bin/activate

# On Windows
.env\Scripts\activate
```

### 2. Process FHIR Files

```bash
python batch_process_fhir.py --input_dir fhir/fhir --output_file combined_claims.tsv
```

This script will:
- Process all FHIR JSON files in the specified directory
- Extract claims information from each file
- Combine all claims into a single TSV file
- Skip already processed files with `--skip_processed` flag

### 3. Train the Model

```bash
python enhanced_train.py --data_file combined_claims.tsv --output_dir ./output

# With torch.compile for faster training (requires PyTorch 2.0+)
python enhanced_train.py --data_file combined_claims.tsv --output_dir ./output --compile
```

The training script includes:
- Data caching for faster loading
- Mixed precision training
- PyTorch compilation (torch.compile) for faster training
- Checkpointing and resuming capability
- Early stopping
- Validation
- TensorBoard logging
- Weights & Biases integration for experiment tracking
- Example inference during training

### 4. Use the Trained Model

After training, you can use the model for inference by:

```python
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

# Load the model and tokenizer
model_path = "path/to/model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForMaskedLM.from_pretrained(model_path)

# Prepare a query with masked tokens
query = "active\tpharmacy\tnormal\t1\t<mask><mask><mask><mask><mask><mask>\t0.45\t2015-08-22\tAMB\t\tcomplete\t0.45\tfemale\t1983-05-07\t41\tBlack or African American\tHispanic or Latino\tMarried\tSpanish\tFalse"
query = query.replace("\t", "<tab>").strip("\n")

# Get model predictions
inputs = tokenizer(query, return_tensors='pt', padding='max_length', truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits

# Helper function to extract predictions
def sample_masked_token(input_ids, logits, tokenizer):
    # Implementation in enhanced_train.py
    pass

predictions = sample_masked_token(inputs.input_ids, logits, tokenizer)
print(f"Predicted missing value: {predictions}")
```

## Shell Script Options

The `run_synthea_bert.sh` script accepts the following parameters:

```
--input_dir      Directory containing FHIR JSON files (default: fhir/fhir)
--output_dir     Output directory for results (default: ./runs)
--batch_size     Training batch size (default: 16)
--num_epochs     Number of training epochs (default: 20)
--learning_rate  Learning rate (default: 0.00005)
--num_workers    Number of worker threads (default: 4)
--skip_processed Skip already processed FHIR files
--disable_wandb  Disable Weights & Biases logging
--compile        Enable torch.compile for faster training (requires PyTorch 2.0+)
--compile_mode   Compilation mode (auto, reduce-overhead, max-autotune)
```

## Advanced Configuration

### Data Processing Options

```
--input_dir: Directory containing FHIR JSON files
--output_file: Path to save the combined TSV file
--skip_processed: Skip already processed files
```

### Training Options

```
# Data arguments
--data_file: Path to the TSV data file
--cache_dir: Directory to cache processed data
--max_length: Maximum sequence length
--validation_split: Proportion of data to use for validation
--force_reload_data: Force reprocessing the data

# Model arguments
--model_name: Base model name or path (default: distilroberta-base)
--resume_from_checkpoint: Path to checkpoint to resume from

# Training arguments
--output_dir: Output directory
--num_epochs: Number of training epochs
--batch_size: Training batch size
--learning_rate: Learning rate
--weight_decay: Weight decay
--warmup_ratio: Proportion of training to perform LR warmup
--early_stopping: Stop after this many epochs without improvement
--inference_interval: Run inference every N epochs

# Performance arguments
--num_workers: Number of worker threads for data loading
--no_cuda: Disable CUDA
--mixed_precision: Use mixed precision training
--compile: Use torch.compile for faster training (requires PyTorch 2.0+)
--compile_mode: Compilation mode for torch.compile (auto, reduce-overhead, max-autotune)

# Logging arguments
--disable_wandb: Disable Weights & Biases logging
```

## Data Structure

The system processes FHIR JSON files and extracts the following information for claims:

- Claim status
- Claim type
- Service information
- Patient demographics
- Dates
- Claim amounts
- Encounter details

## Model Performance

The model is trained to predict masked fields in claims data, particularly medication information. Performance can be evaluated using:

1. Validation loss during training
2. Perplexity
3. Example inference tests with known medications

## Monitoring Training

### TensorBoard

You can monitor training with TensorBoard:

```bash
# Activate the virtual environment first
source .env/bin/activate

# Then run TensorBoard
tensorboard --logdir ./output/tensorboard
```

This will show:
- Loss curves
- Learning rate
- Perplexity
- Example predictions
- Token processing statistics

### Weights & Biases

The pipeline automatically integrates with Weights & Biases (wandb) for experiment tracking:

```bash
# Training will automatically log to the "synthea_train" project
# You can disable wandb logging with:
./run_synthea_bert.sh --disable_wandb
```

Weights & Biases provides:
- Real-time training metrics visualization
- Experiment comparison
- Hyperparameter tracking
- Training run history
- Token processing statistics
- System resource usage monitoring

On first use, you'll need to log in with your wandb account:
```bash
wandb login
```

## Performance Optimization

### torch.compile (PyTorch 2.0+)

The pipeline supports PyTorch's compilation feature for faster training:

```bash
# Enable compilation with default mode
./run_synthea_bert.sh --compile

# Use a specific compilation mode
./run_synthea_bert.sh --compile --compile_mode reduce-overhead
```

Compilation modes:
- `auto` (default): Let PyTorch choose the best mode
- `reduce-overhead`: Minimize compilation overhead
- `max-autotune`: Maximize performance through autotuning

Benefits:
- Up to 2x faster training speed
- No changes to model architecture required
- Optimized execution graph on both CPU and GPU

Note that torch.compile requires PyTorch 2.0 or newer.

## Scaling Tips

For large datasets:

1. Use `--skip_processed` to resume interrupted data processing
2. Enable `--mixed_precision` for faster training
3. Enable `--compile` if using PyTorch 2.0+ for further speedups
4. Adjust `--num_workers` based on your CPU
5. Use `--batch_size` to maximize GPU utilization
6. If out of memory, reduce batch size and use gradient accumulation
7. Use `--resume_from_checkpoint` to continue training from interruptions

## License

[MIT License] 