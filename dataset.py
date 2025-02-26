from torch.utils.data import Dataset
import os
import logging
import time
import atexit
import shutil
import glob
import gc
import signal
import tempfile
import weakref
import threading

logger = logging.getLogger(__name__)

# Global registry to track open file handles for cleanup
_OPEN_FILES = weakref.WeakSet()

# Persistent cleanup daemon
class CleanupDaemon:
    def __init__(self, interval=5):
        self.interval = interval
        self.daemon = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.running = True
        self.daemon.start()
        logger.debug("Cleanup daemon thread started")
        
    def _cleanup_loop(self):
        while self.running:
            try:
                self.cleanup_resources()
            except Exception as e:
                logger.debug(f"Error in cleanup daemon: {e}")
            time.sleep(self.interval)
    
    def stop(self):
        self.running = False
        
    def cleanup_resources(self):
        """Cleanup resources - called periodically by the daemon thread"""
        # First, close any lingering file handles
        for file_obj in list(_OPEN_FILES):
            try:
                if hasattr(file_obj, 'close') and not file_obj.closed:
                    file_obj.close()
            except Exception as e:
                logger.debug(f"Error closing file handle: {e}")
        
        # Force garbage collection to release file handles
        gc.collect()
        
        # Now clean up temp directories more aggressively
        try:
            # Define common temp dirs patterns
            temp_patterns = ['/tmp/pymp-*', tempfile.gettempdir() + '/pymp-*']
            
            for pattern in temp_patterns:
                try:
                    dirs = glob.glob(pattern)
                    for d in dirs:
                        try:
                            if os.path.exists(d):
                                # First try killing any processes that might be keeping the directory open
                                self._force_clean_directory(d)
                                
                                # Try more aggressive cleanup
                                shutil.rmtree(d, ignore_errors=True)
                                
                                # If still exists, try deeper cleanup
                                if os.path.exists(d):
                                    # Make all contents writable to ensure they can be deleted
                                    for root, dirs_list, files in os.walk(d, topdown=False):
                                        for name in files:
                                            try:
                                                file_path = os.path.join(root, name)
                                                os.chmod(file_path, 0o777)
                                            except Exception:
                                                pass
                                        for name in dirs_list:
                                            try:
                                                dir_path = os.path.join(root, name)
                                                os.chmod(dir_path, 0o777)
                                            except Exception:
                                                pass
                                                
                                    # Try once more with force
                                    shutil.rmtree(d, ignore_errors=True)
                        except Exception as e:
                            logger.debug(f"Error removing directory {d}: {e}")
                except Exception as e:
                    logger.debug(f"Error processing pattern {pattern}: {e}")
        except Exception as e:
            logger.debug(f"Error in temp directory cleanup: {e}")
    
    def _force_clean_directory(self, directory):
        """More aggressive approach to ensure directory can be removed"""
        try:
            # On Linux, check if we can find processes with open files in this directory
            if os.name == 'posix':
                try:
                    # Try using lsof to find processes with open files
                    import subprocess
                    result = subprocess.run(['lsof', '+D', directory], 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE,
                                           timeout=2)
                    if result.returncode == 0:
                        # Parse output to find process IDs
                        output = result.stdout.decode('utf-8')
                        for line in output.split('\n')[1:]:  # Skip header
                            if line.strip():
                                parts = line.split()
                                if len(parts) > 1:
                                    try:
                                        pid = int(parts[1])
                                        # Try to terminate the process gently
                                        os.kill(pid, signal.SIGTERM)
                                        logger.debug(f"Sent SIGTERM to process {pid} holding files in {directory}")
                                    except (ValueError, ProcessLookupError, PermissionError):
                                        pass
                except Exception:
                    pass
        except Exception:
            pass

# Create a global cleanup daemon
_cleanup_daemon = CleanupDaemon(interval=5)

# Improved cleanup function for temporary directories
def cleanup_temp_dirs():
    """Safely clean up any lingering temporary directories and open file handles."""
    if hasattr(_cleanup_daemon, 'cleanup_resources'):
        _cleanup_daemon.cleanup_resources()
    else:
        logger.warning("Cleanup daemon not initialized properly")

# Register cleanup function to run at exit
atexit.register(cleanup_temp_dirs)

# Register signal handlers more safely
def signal_handler(signum, frame):
    cleanup_temp_dirs()
    # Call the default handler
    original_handlers[signum](signum, frame)

# Store original handlers
original_handlers = {}
for sig in [signal.SIGTERM, signal.SIGINT]:
    try:
        original_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, signal_handler)
    except (ValueError, TypeError):
        pass  # Some signals might not be available on all platforms

class ClaimsMaskedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []

        # Process all texts
        for text in texts:
            # Replace actual tabs with the <tab> token
            text = text.replace('\t', '<tab>')

            masked_examples, original_texts = self.mask_row(text.strip("\n"))

            for masked_text, original_text in zip(masked_examples, original_texts):
                # Tokenize masked input
                masked_inputs = tokenizer(masked_text,
                                        max_length=max_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt')

                # Tokenize target (original text)
                target_inputs = tokenizer(original_text,
                                        max_length=max_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt')

                self.inputs.append({
                    'input_ids': masked_inputs['input_ids'].squeeze(),
                    'attention_mask': masked_inputs['attention_mask'].squeeze()
                })

                self.targets.append(target_inputs['input_ids'].squeeze())

    def mask_row(self, text):
        """
        More efficient implementation to mask each column value separately.

        Args:
            text: A string with column values separated by <tab> tokens

        Returns:
            masked_examples: List of strings where each has one column value masked
            original_texts: List of the original text repeated for each example
        """
        # Split by <tab> to get column values
        columns = text.split('<tab>')
        masked_examples = []

        # For each column, create a version with that column masked
        for i in range(len(columns)):
            if columns[i].strip() == '':  # Skip empty columns
                continue

            # Create a copy of columns and mask the target column
            masked_columns = columns.copy()

            # Replace the column with mask tokens (one per token in the column)
            tokens = self.tokenizer.tokenize(columns[i])
            masked_columns[i] = ''.join(['<mask>'] * len(tokens))

            # Join back with <tab> tokens
            masked_example = '<tab>'.join(masked_columns)
            masked_examples.append(masked_example)

        # Return the masked examples and the original text repeated
        return masked_examples, [text] * len(masked_examples)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]['input_ids']
        attention_mask = self.inputs[idx]['attention_mask']
        labels = self.targets[idx].clone()

        # Create a mask where only the masked tokens contribute to loss
        mask_positions = (input_ids == self.tokenizer.mask_token_id)

        # Set labels to -100 for all positions EXCEPT the masked ones
        labels[~mask_positions] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class StreamingClaimsMaskedDataset(Dataset):
    """
    Memory-efficient dataset that doesn't load all texts at once.
    Instead, it indexes the file and processes lines on-demand.
    """
    def __init__(self, file_path, tokenizer, max_length=128, precompute_size=False):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.line_offsets = []
        self.item_to_line_map = []
        self.cached_inputs = {}  # Small cache for recently accessed items
        self.cache_size = 1000  # Maximum number of cached items
        
        # Index the file for fast random access
        self._index_file()
        
        # Precompute dataset size (slower but gives accurate size upfront)
        self.num_examples_per_line = []
        self.total_examples = 0
        
        if precompute_size:
            logger.info("Precomputing dataset size (this may take a while)...")
            self._precompute_size()
        else:
            # Estimate size based on the first 100 lines
            self._estimate_size()
    
    def _index_file(self):
        """Create an index of line positions in the file for random access."""
        logger.info(f"Indexing file: {self.file_path}")
        start_time = time.time()
        self.line_offsets = [0]  # Start of the first line
        
        # Check if we've already indexed this file before
        index_cache_path = f"{self.file_path}.index"
        if os.path.exists(index_cache_path):
            try:
                logger.info(f"Loading existing file index from {index_cache_path}")
                with open(index_cache_path, 'r') as f:
                    self.line_offsets = [int(line.strip()) for line in f]
                logger.info(f"Loaded index with {len(self.line_offsets):,} lines")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Rebuilding...")
        
        # Build index from scratch
        with open(self.file_path, 'r') as f:
            while f.readline():
                self.line_offsets.append(f.tell())
        
        # Remove the last entry which is EOF
        self.line_offsets.pop()
        
        # Save index for future use
        try:
            with open(index_cache_path, 'w') as f:
                for offset in self.line_offsets:
                    f.write(f"{offset}\n")
            logger.info(f"Saved file index to {index_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save index: {e}")
        
        index_time = time.time() - start_time
        logger.info(f"Indexed {len(self.line_offsets):,} lines in {index_time:.2f} seconds")
    
    def _estimate_size(self):
        """Estimate dataset size based on a sample of the first 100 lines."""
        start_time = time.time()
        sample_size = min(100, len(self.line_offsets))
        total_examples = 0
        
        with open(self.file_path, 'r') as f:
            for i in range(sample_size):
                f.seek(self.line_offsets[i])
                line = f.readline().strip()
                if not line:
                    continue
                
                # Count tabs to estimate columns (add 1 because n tabs = n+1 columns)
                num_columns = line.count('\t') + 1
                # Each column creates one example, but skip empty columns
                line = line.replace('\t', '<tab>')
                columns = line.split('<tab>')
                num_examples = sum(1 for col in columns if col.strip())
                
                total_examples += num_examples
        
        avg_examples_per_line = total_examples / sample_size
        estimated_total = int(avg_examples_per_line * len(self.line_offsets))
        
        # Use this as our estimate
        self.total_examples = estimated_total
        estimate_time = time.time() - start_time
        logger.info(f"Estimated dataset size: {estimated_total:,} examples from {len(self.line_offsets):,} lines")
        logger.info(f"Average examples per line: {avg_examples_per_line:.2f}")
        logger.info(f"Size estimation completed in {estimate_time:.2f} seconds")
    
    def _precompute_size(self):
        """Precisely compute the total number of examples by scanning the entire file."""
        total_examples = 0
        
        with open(self.file_path, 'r') as f:
            for i, offset in enumerate(self.line_offsets):
                if i % 10000 == 0:
                    logger.info(f"Precomputing size - processed {i}/{len(self.line_offsets)} lines")
                
                f.seek(offset)
                line = f.readline().strip()
                if not line:
                    self.num_examples_per_line.append(0)
                    continue
                
                line = line.replace('\t', '<tab>')
                columns = line.split('<tab>')
                num_examples = sum(1 for col in columns if col.strip())
                
                self.num_examples_per_line.append(num_examples)
                
                # Create mapping from example index to line index
                for _ in range(num_examples):
                    self.item_to_line_map.append(i)
                
                total_examples += num_examples
        
        self.total_examples = total_examples
        logger.info(f"Exact dataset size: {total_examples} examples from {len(self.line_offsets)} lines")
    
    def _process_line(self, line_idx):
        """Process a single line from the file and return all examples."""
        if line_idx >= len(self.line_offsets):
            raise IndexError(f"Line index {line_idx} out of range (max {len(self.line_offsets)-1})")
        
        with open(self.file_path, 'r') as f:
            f.seek(self.line_offsets[line_idx])
            line = f.readline().strip()
        
        if not line:
            return []
        
        # Replace tabs with <tab> token
        line = line.replace('\t', '<tab>')
        
        # Mask each column and create examples
        masked_examples, original_texts = self._mask_row(line)
        
        examples = []
        for masked_text, original_text in zip(masked_examples, original_texts):
            # Tokenize masked input
            masked_inputs = self.tokenizer(masked_text,
                                    max_length=self.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt')
            
            # Tokenize target (original text)
            target_inputs = self.tokenizer(original_text,
                                    max_length=self.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt')
            
            input_ids = masked_inputs['input_ids'].squeeze()
            attention_mask = masked_inputs['attention_mask'].squeeze()
            labels = target_inputs['input_ids'].squeeze().clone()
            
            # Create a mask where only the masked tokens contribute to loss
            mask_positions = (input_ids == self.tokenizer.mask_token_id)
            
            # Set labels to -100 for all positions EXCEPT the masked ones
            labels[~mask_positions] = -100
            
            examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        
        return examples
    
    def _mask_row(self, text):
        """
        Mask each column value separately.
        
        Args:
            text: A string with column values separated by <tab> tokens
            
        Returns:
            masked_examples: List of strings where each has one column value masked
            original_texts: List of the original text repeated for each example
        """
        # Split by <tab> to get column values
        columns = text.split('<tab>')
        masked_examples = []
        
        # For each column, create a version with that column masked
        for i in range(len(columns)):
            if columns[i].strip() == '':  # Skip empty columns
                continue
            
            # Create a copy of columns and mask the target column
            masked_columns = columns.copy()
            
            # Replace the column with mask tokens (one per token in the column)
            tokens = self.tokenizer.tokenize(columns[i])
            masked_columns[i] = ''.join(['<mask>'] * len(tokens))
            
            # Join back with <tab> tokens
            masked_example = '<tab>'.join(masked_columns)
            masked_examples.append(masked_example)
        
        # Return the masked examples and the original text repeated
        return masked_examples, [text] * len(masked_examples)
    
    def get_line_based_item(self, line_idx, column_idx):
        """Get a specific item based on line and column indices."""
        line_examples = self._process_line(line_idx)
        if column_idx >= len(line_examples):
            raise IndexError(f"Column index {column_idx} out of range for line {line_idx}")
        return line_examples[column_idx]
    
    def __len__(self):
        return self.total_examples
    
    def __getitem__(self, idx):
        """Get an item by its index in the dataset."""
        # Check cache first
        if idx in self.cached_inputs:
            return self.cached_inputs[idx]
        
        # If we've precomputed the mapping, use it
        if self.item_to_line_map:
            line_idx = self.item_to_line_map[idx]
            examples = self._process_line(line_idx)
            
            # Find which example within the line
            item_count = 0
            for i in range(line_idx):
                item_count += self.num_examples_per_line[i]
            
            example_idx = idx - item_count
            item = examples[example_idx]
        else:
            # Otherwise, scan through the file
            line_idx = 0
            item_count = 0
            
            while line_idx < len(self.line_offsets):
                examples = self._process_line(line_idx)
                if item_count + len(examples) > idx:
                    # The item is in this line
                    example_idx = idx - item_count
                    item = examples[example_idx]
                    break
                
                item_count += len(examples)
                line_idx += 1
            else:
                raise IndexError(f"Index {idx} out of range for dataset of size {self.__len__()}")
        
        # Update cache
        self.cached_inputs[idx] = item
        
        # Manage cache size
        if len(self.cached_inputs) > self.cache_size:
            # Remove the oldest entries (20% of cache size)
            keys_to_remove = list(self.cached_inputs.keys())[:int(self.cache_size * 0.2)]
            for key in keys_to_remove:
                del self.cached_inputs[key]
        
        return item

class SequentialStreamingClaimsMaskedDataset(Dataset):
    """
    Memory-efficient dataset that processes data sequentially, one line at a time.
    This implementation is optimized for pure sequential access without random access overhead.
    It tracks the current position and processes data in batches.
    """
    def __init__(self, file_path, tokenizer, max_length=128, batch_size=16):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Count lines for epoch tracking (without verbose logging)
        self.line_count = self._count_lines()
        
        # Set up file objects for reading - use a safer approach with tracking
        self.file = open(file_path, 'r')
        # Register file handle for cleanup
        _OPEN_FILES.add(self.file)
        
        self.current_line = 0
        self.current_epoch = 0
        
        # Sample a few lines to estimate examples per line for size estimation
        self.estimated_examples_per_line = self._estimate_examples_per_line()
        self.total_examples = int(self.line_count * self.estimated_examples_per_line)
        
        # Batch processing state
        self.current_batch = []
        self.batch_position = 0
    
    def __del__(self):
        """Ensure file resources are closed when this object is garbage collected."""
        self.close()
    
    def close(self):
        """Explicitly close all file resources."""
        try:
            if hasattr(self, 'file') and self.file and not self.file.closed:
                self.file.close()
        except Exception as e:
            logger.debug(f"Error closing file in dataset: {e}")
    
    def _count_lines(self):
        """Count lines in the file efficiently."""
        # For very large files, use a faster method
        file_size = os.path.getsize(self.file_path)
        if file_size > 100 * 1024 * 1024:  # If > 100MB
            # Sample the file to get average line length, then estimate
            sample_size = 1000
            with open(self.file_path, 'r') as f:
                sample_lines = [next(f) for _ in range(sample_size) if _ < sample_size]
            
            avg_line_length = sum(len(line) for line in sample_lines) / len(sample_lines)
            estimated_lines = int(file_size / avg_line_length)
            return estimated_lines
        else:
            # For smaller files, count directly
            with open(self.file_path, 'r') as f:
                count = sum(1 for _ in f)
            return count
    
    def _estimate_examples_per_line(self):
        """Estimate how many examples (masked items) we get per line on average."""
        # Reset file position
        self.file.seek(0)
        
        # Sample a few lines to get an estimate
        sample_size = min(100, self.line_count)
        total_examples = 0
        
        for _ in range(sample_size):
            line = self.file.readline().strip()
            if not line:
                continue
            
            # Count non-empty columns (potential mask targets)
            line = line.replace('\t', '<tab>')
            columns = line.split('<tab>')
            num_examples = sum(1 for col in columns if col.strip())
            
            total_examples += num_examples
        
        # Reset file position
        self.file.seek(0)
        
        # Calculate average examples per line
        avg_examples = total_examples / sample_size if sample_size > 0 else 0
        return avg_examples
    
    def _get_next_batch(self, batch_size):
        """Get the next batch of examples by processing lines sequentially."""
        # Process lines until we have enough examples for a batch
        examples = []
        lines_processed = 0
        
        while len(examples) < batch_size and lines_processed < self.line_count:
            # Read the next line
            line = self.file.readline().strip()
            if not line:
                # If we reach EOF, wrap around to the beginning for the next epoch
                # Close and reopen file handle to prevent resource leaks
                self.file.close()
                self.file = open(self.file_path, 'r')
                # Register the new file handle
                _OPEN_FILES.add(self.file)
                
                self.current_epoch += 1
                # No more logger.info for epoch starts - tqdm will handle this
                line = self.file.readline().strip()
            
            # Process the line
            self.current_line += 1
            lines_processed += 1
            
            # Skip empty lines
            if not line:
                continue
            
            # Replace tabs with <tab> token
            line = line.replace('\t', '<tab>')
            
            # Create masked examples from this line
            masked_examples, original_texts = self._mask_row(line)
            
            # Process each masked example
            for masked_text, original_text in zip(masked_examples, original_texts):
                # Tokenize masked input
                masked_inputs = self.tokenizer(masked_text,
                                             max_length=self.max_length,
                                             padding='max_length',
                                             truncation=True,
                                             return_tensors='pt')
                
                # Tokenize target (original text)
                target_inputs = self.tokenizer(original_text,
                                             max_length=self.max_length,
                                             padding='max_length',
                                             truncation=True,
                                             return_tensors='pt')
                
                input_ids = masked_inputs['input_ids'].squeeze()
                attention_mask = masked_inputs['attention_mask'].squeeze()
                labels = target_inputs['input_ids'].squeeze().clone()
                
                # Create a mask where only the masked tokens contribute to loss
                mask_positions = (input_ids == self.tokenizer.mask_token_id)
                
                # Set labels to -100 for all positions EXCEPT the masked ones
                labels[~mask_positions] = -100
                
                examples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })
                
                # Stop if we have enough examples
                if len(examples) >= batch_size:
                    break
        
        # Remove the periodic logging - tqdm will handle progress display
        return examples
    
    def _mask_row(self, text):
        """
        Mask each column value separately.
        
        Args:
            text: A string with column values separated by <tab> tokens
            
        Returns:
            masked_examples: List of strings where each has one column value masked
            original_texts: List of the original text repeated for each example
        """
        # Split by <tab> to get column values
        columns = text.split('<tab>')
        masked_examples = []
        
        # For each column, create a version with that column masked
        for i in range(len(columns)):
            if columns[i].strip() == '':  # Skip empty columns
                continue
            
            # Create a copy of columns and mask the target column
            masked_columns = columns.copy()
            
            # Replace the column with mask tokens (one per token in the column)
            tokens = self.tokenizer.tokenize(columns[i])
            masked_columns[i] = ''.join(['<mask>'] * len(tokens))
            
            # Join back with <tab> tokens
            masked_example = '<tab>'.join(masked_columns)
            masked_examples.append(masked_example)
        
        # Return the masked examples and the original text repeated
        return masked_examples, [text] * len(masked_examples)
    
    def __len__(self):
        """Return an estimate of the total number of examples."""
        return self.total_examples
    
    def __getitem__(self, idx):
        """Get next item from the dataset - sequential access ignoring the index."""
        # Check if we need to generate a new batch
        if not self.current_batch or self.batch_position >= len(self.current_batch):
            self.current_batch = self._get_next_batch(self.batch_size)
            self.batch_position = 0
            
            # If we still couldn't get any examples, there's a problem with the file
            if not self.current_batch:
                raise RuntimeError(f"Failed to load examples from {self.file_path}")
        
        # Get the next example from the current batch
        item = self.current_batch[self.batch_position]
        self.batch_position += 1
        
        return item
    
    def reset(self):
        """Reset the dataset to the beginning of the file."""
        # Close existing file handle first if it exists
        if hasattr(self, 'file') and self.file and not self.file.closed:
            self.file.close()
        
        # Open a fresh file handle
        self.file = open(self.file_path, 'r')
        # Register file handle for cleanup
        _OPEN_FILES.add(self.file)
        
        self.current_line = 0
        self.current_epoch = 0
        self.current_batch = []
        self.batch_position = 0