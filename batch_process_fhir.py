import os
import glob
import json
import pandas as pd
from tqdm import tqdm
from data_processing import parse_fhir_bundle
from pathlib import Path
import argparse
import logging

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

def process_all_fhir_files(input_dir, output_file, skip_processed=False):
    """
    Process all FHIR JSON files in the input directory and save the combined claims data to a TSV file.
    
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
    
    # Initialize an empty list to store all the claim DataFrames
    all_claims_dfs = []
    
    # Process each JSON file
    for json_file in tqdm(json_files, desc="Processing FHIR files"):
        file_name = os.path.basename(json_file)
        temp_file = os.path.join(temp_dir, f"{file_name}.processed")
        
        # Skip if already processed and skip_processed is True
        if skip_processed and os.path.exists(temp_file):
            logger.info(f"Skipping already processed file: {file_name}")
            
            # Load the previously processed file
            temp_tsv = os.path.join(temp_dir, f"{file_name}.tsv")
            if os.path.exists(temp_tsv):
                try:
                    claims_df = pd.read_csv(temp_tsv, sep='\t')
                    # Filter to include only the required columns if they exist
                    required_columns = [
                        "claim_type", "days_of_service", "product_service_display",
                        "claim_amount", "encounter_class", "encounter_reason",
                        "gender", "age"
                    ]
                    existing_columns = [col for col in required_columns if col in claims_df.columns]
                    claims_df = claims_df[existing_columns]
                    all_claims_dfs.append(claims_df)
                    logger.info(f"Loaded {len(claims_df)} claims from {temp_tsv}")
                except Exception as e:
                    logger.error(f"Error loading {temp_tsv}: {e}")
            
            continue
        
        try:
            # Parse the FHIR Bundle
            logger.info(f"Processing {file_name}")
            dfs = parse_fhir_bundle(json_file)
            
            # Check if claims data was extracted
            if "claims" in dfs and not dfs["claims"].empty:
                claims_df = dfs["claims"]
                
                # Filter to include only the required columns if they exist
                required_columns = [
                    "claim_type", "days_of_service", "product_service_display",
                    "claim_amount", "encounter_class", "encounter_reason",
                    "gender", "age"
                ]
                existing_columns = [col for col in required_columns if col in claims_df.columns]
                claims_df = claims_df[existing_columns]
                
                all_claims_dfs.append(claims_df)
                
                # Save individual file's claims to temp directory for incremental processing
                temp_tsv = os.path.join(temp_dir, f"{file_name}.tsv")
                claims_df.to_csv(temp_tsv, sep='\t', index=False)
                logger.info(f"Extracted {len(claims_df)} claims from {file_name}")
                
                # Create an empty file to mark this file as processed
                with open(temp_file, 'w') as f:
                    pass
            else:
                logger.warning(f"No claims data found in {file_name}")
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
    
    # Combine all DataFrames
    if all_claims_dfs:
        combined_df = pd.concat(all_claims_dfs, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} total claims from {len(all_claims_dfs)} files")
        
        # Remove duplicate rows
        combined_df = combined_df.drop_duplicates()
        logger.info(f"After removing duplicates: {len(combined_df)} claims")
        
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