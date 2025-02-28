import os
import glob
import json
import pandas as pd
from tqdm import tqdm
from data_processing import parse_fhir_bundle
from pathlib import Path
import argparse
import logging
import multiprocessing
from functools import partial
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fhir_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_single_file(json_file, temp_dir, required_columns):
    """
    Process a single FHIR JSON file and return the claims DataFrame.
    
    Args:
        json_file (str): Path to the FHIR JSON file
        temp_dir (str): Directory to save temporary processed files
        required_columns (list): List of columns to keep in the DataFrame
        
    Returns:
        pd.DataFrame or None: Claims DataFrame if successful, None otherwise
    """
    file_name = os.path.basename(json_file)
    temp_file = os.path.join(temp_dir, f"{file_name}.processed")
    temp_tsv = os.path.join(temp_dir, f"{file_name}.tsv")
    
    try:
        # Parse the FHIR Bundle
        dfs = parse_fhir_bundle(json_file)
        
        # Check if claims data was extracted
        if "claims" in dfs and not dfs["claims"].empty:
            claims_df = dfs["claims"]
            
            # Filter to include only the required columns if they exist
            existing_columns = [col for col in required_columns if col in claims_df.columns]
            claims_df = claims_df[existing_columns]
            
            # Save individual file's claims to temp directory for incremental processing
            claims_df.to_csv(temp_tsv, sep='\t', index=False)
            
            # Create an empty file to mark this file as processed
            with open(temp_file, 'w') as f:
                pass
                
            return claims_df
        else:
            logger.warning(f"No claims data found in {file_name}")
            return None
    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")
        return None

def load_processed_file(temp_tsv, required_columns):
    """
    Load a previously processed temp file.
    
    Args:
        temp_tsv (str): Path to the temp TSV file
        required_columns (list): List of columns to keep in the DataFrame
        
    Returns:
        pd.DataFrame or None: Claims DataFrame if successful, None otherwise
    """
    try:
        claims_df = pd.read_csv(temp_tsv, sep='\t')
        # Filter to include only the required columns if they exist
        existing_columns = [col for col in required_columns if col in claims_df.columns]
        claims_df = claims_df[existing_columns]
        return claims_df
    except Exception as e:
        logger.error(f"Error loading {temp_tsv}: {e}")
        return None

def process_all_fhir_files(input_dir, output_file, skip_processed=False):
    """
    Process all FHIR JSON files in the input directory and save the combined claims data to a TSV file.
    Uses multiprocessing to parallelize the processing.
    
    Args:
        input_dir (str): Directory containing FHIR JSON files
        output_file (str): Path to save the combined TSV file
        skip_processed (bool): Skip files that have already been processed (based on temp files)
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a temp directory for individual processed files
    temp_dir = os.path.join(output_dir, "temp_processed")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Get all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Define the columns we want to keep
    required_columns = [
        "claim_type", "days_of_service", "product_service_display",
        "claim_amount", "encounter_class", "encounter_reason",
        "gender", "age"
    ]
    
    # Files to process
    files_to_process = []
    previously_processed_dfs = []
    
    # Identify which files to process and which to load from temp
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        temp_file = os.path.join(temp_dir, f"{file_name}.processed")
        temp_tsv = os.path.join(temp_dir, f"{file_name}.tsv")
        
        if skip_processed and os.path.exists(temp_file) and os.path.exists(temp_tsv):
            # Load previously processed file
            logger.info(f"Loading previously processed file: {file_name}")
            claims_df = load_processed_file(temp_tsv, required_columns)
            if claims_df is not None:
                previously_processed_dfs.append(claims_df)
        else:
            # Add to list of files to process
            files_to_process.append(json_file)
    
    logger.info(f"Loaded {len(previously_processed_dfs)} previously processed files")
    logger.info(f"Processing {len(files_to_process)} new files with multiprocessing")
    
    # Use multiprocessing to process files in parallel
    if files_to_process:
        # Get number of cores
        num_cores = multiprocessing.cpu_count()
        logger.info(f"Using {num_cores} CPU cores for parallel processing")
        
        # Create a partial function with fixed arguments
        process_file_partial = partial(
            process_single_file, 
            temp_dir=temp_dir, 
            required_columns=required_columns
        )
        
        # Create a pool of workers and process files in parallel
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Process files in parallel and show progress with tqdm
            results = list(tqdm(
                pool.imap(process_file_partial, files_to_process),
                total=len(files_to_process),
                desc="Processing FHIR files"
            ))
        
        # Filter out None results
        processed_dfs = [df for df in results if df is not None]
        logger.info(f"Successfully processed {len(processed_dfs)} new files")
        
        # Combine with previously processed DataFrames
        all_claims_dfs = previously_processed_dfs + processed_dfs
    else:
        all_claims_dfs = previously_processed_dfs
    
    # Combine all DataFrames
    if all_claims_dfs:
        combined_df = pd.concat(all_claims_dfs, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} total claims from {len(all_claims_dfs)} files")
        
        # Remove duplicate rows
        combined_df = combined_df.drop_duplicates()
        logger.info(f"After removing duplicates: {len(combined_df)} claims")
        
        # Shuffle the data for randomization
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info("Shuffled data for randomization")
        
        # Save to TSV file
        combined_df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"Saved combined claims data to {output_file}")
        
        return len(combined_df)
    else:
        logger.warning("No claims data found in any of the processed files")
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FHIR JSON files and extract claims data")
    parser.add_argument("--input_dir", type=str, default="fhir/fhir", help="Directory containing FHIR JSON files")
    parser.add_argument("--output_file", type=str, default="combined_claims.tsv", help="Output TSV file path")
    parser.add_argument("--skip_processed", action="store_true", help="Skip already processed files")
    
    args = parser.parse_args()
    
    process_all_fhir_files(args.input_dir, args.output_file, args.skip_processed) 