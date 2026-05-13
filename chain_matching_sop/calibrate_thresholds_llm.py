#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for LLM-based threshold calibration
Calls LLM API to judge whether source texts match chain segment definitions
"""

import json
import time
import requests
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.io_utils import load_json, save_json
from prompt.threshold_llm import base_prompt
from config.data_config import (
    LLM_API_URL,
    LLM_API_KEY,
    LLM_MODEL_NAME,
    LLM_RECORDS_PER_BATCH,
    LLM_MAX_CONCURRENT,
    LLM_RETRY_WAIT,
    LLM_MAX_RETRIES,
    LLM_HTTP_TIMEOUT,
    LLM_SIMILARITY_SAMPLES_PATH,
    LLM_CHAIN_DEFINITIONS_PATH,
    LLM_OUTPUT_PATH,
    LLM_SAVE_INTERVAL,
)


# ============================================================================
# Configuration Parameters
# ============================================================================
# All configuration parameters are now imported from config.data_config
# to enable centralized configuration management.
# See config/data_config.py for detailed parameter descriptions and adjustments.

# Create local aliases for imported configuration parameters
# to maintain compatibility with existing code
API_URL = LLM_API_URL
API_KEY = LLM_API_KEY
MODEL_NAME = LLM_MODEL_NAME
RECORDS_PER_BATCH = LLM_RECORDS_PER_BATCH
MAX_CONCURRENT = LLM_MAX_CONCURRENT
RETRY_WAIT = LLM_RETRY_WAIT
MAX_RETRIES = LLM_MAX_RETRIES
HTTP_TIMEOUT = LLM_HTTP_TIMEOUT
SIMILARITY_SAMPLES_PATH = LLM_SIMILARITY_SAMPLES_PATH
CHAIN_DEFINITIONS_PATH = LLM_CHAIN_DEFINITIONS_PATH
OUTPUT_PATH = LLM_OUTPUT_PATH

# Thread-safe lock for file writing
file_lock = threading.Lock()


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_similarity_samples(file_path: Path) -> Dict[str, Dict[str, List]]:
    """
    Load similarity samples from JSON file.

    Args:
        file_path: Path to similarity_samples.json

    Returns:
        Dictionary with structure: {chain_name: {interval: [[id, source_text], ...]}}
    """
    print("=" * 70)
    print("Loading similarity samples...")
    print("=" * 70)
    
    data = load_json(file_path)
    print(f"Loaded {len(data)} chain segments")
    print()
    
    return data


def load_chain_definitions(file_path: Path) -> Dict[str, str]:
    """
    Load chain definitions from JSON file.
    
    Args:
        file_path: Path to chain_definitions JSON file
    
    Returns:
        Dictionary mapping chain names to their definitions
    """
    print("=" * 70)
    print("Loading chain definitions...")
    print("=" * 70)
    
    definitions_list = load_json(file_path)
    
    # Convert list to dictionary: {name: definition}
    definitions_dict = {}
    for item in definitions_list:
        if 'name' in item and 'definition' in item:
            definitions_dict[item['name']] = item['definition']
    
    print(f"Loaded {len(definitions_dict)} chain definitions")
    print()
    
    return definitions_dict


# ============================================================================
# Prompt Building Functions
# ============================================================================

def format_source_list(records: List[List]) -> str:
    """
    Format source text records into the required string format.
    Each line: [ID] source_text

    Args:
        records: List of [id, source_text] pairs

    Returns:
        Formatted string with one record per line
    """
    lines = []
    for record in records:
        record_id = record[0]
        source_text = record[1]
        lines.append(f"[{record_id}] {source_text}")

    return "\n".join(lines)


def build_prompt(chain_name: str, chain_definition: str, records: List[List]) -> str:
    """
    Build the complete prompt for LLM by filling in the base_prompt template.

    Args:
        chain_name: Name of the chain segment
        chain_definition: Definition of the chain segment
        records: List of [id, source_text] pairs

    Returns:
        Complete prompt string
    """
    formatted_sources = format_source_list(records)
    prompt = base_prompt.format(chain_name, chain_definition, formatted_sources)
    return prompt


# ============================================================================
# LLM API Calling Functions
# ============================================================================

def call_llm_api(prompt: str, retry_count: int = 0) -> Dict[str, Any]:
    """
    Call LLM API with the given prompt.
    
    Args:
        prompt: The prompt text to send
        retry_count: Current retry attempt number
    
    Returns:
        Parsed JSON response from LLM
    
    Raises:
        Exception: If all retry attempts fail
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        
        if response.status_code != 200:
            raise Exception(f"API returned status code {response.status_code}: {response.text}")
        
        result = response.json()
        
        # Extract content from response
        content = result["choices"][0]["message"]["content"]
        
        # Parse the JSON array from content
        parsed_results = json.loads(content)
        
        return parsed_results
        
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"  ⚠ Request failed (attempt {retry_count + 1}/{MAX_RETRIES}): {str(e)}")
            print(f"  ⏳ Waiting {RETRY_WAIT} seconds before retry...")
            time.sleep(RETRY_WAIT)
            return call_llm_api(prompt, retry_count + 1)
        else:
            raise Exception(f"Failed after {MAX_RETRIES} retries: {str(e)}")


# ============================================================================
# Batch Processing Functions
# ============================================================================

class BatchJob:
    """Represents a single batch job with its metadata"""
    def __init__(self, chain_name: str, chain_definition: str, interval: str, 
                 batch_records: List[List], batch_idx: int, interval_total: int):
        self.chain_name = chain_name
        self.chain_definition = chain_definition
        self.interval = interval
        self.batch_records = batch_records
        self.batch_idx = batch_idx
        self.interval_total = interval_total
        
    def __repr__(self):
        return f"BatchJob({self.chain_name}, {self.interval}, batch {self.batch_idx})"


def process_single_batch(job: BatchJob) -> Tuple[str, str, int, List[Dict]]:
    """
    Process a single batch of records by calling LLM API.
    
    Args:
        job: BatchJob object containing all necessary information
    
    Returns:
        Tuple of (chain_name, interval, batch_idx, parsed_results)
    """
    prompt = build_prompt(job.chain_name, job.chain_definition, job.batch_records)
    results = call_llm_api(prompt)
    return (job.chain_name, job.interval, job.batch_idx, results)


def prepare_all_batch_jobs(
    similarity_samples: Dict[str, Dict[str, List]],
    chain_definitions: Dict[str, str]
) -> List[BatchJob]:
    """
    Prepare all batch jobs from all chains and intervals.
    
    Args:
        similarity_samples: Dictionary of similarity samples
        chain_definitions: Dictionary of chain definitions
    
    Returns:
        List of all BatchJob objects
    """
    all_jobs = []
    
    print("=" * 70)
    print("Preparing batch jobs...")
    print("=" * 70)
    
    for chain_name, intervals_dict in similarity_samples.items():
        # Skip if no definition
        if chain_name not in chain_definitions:
            print(f"⚠ Warning: No definition found for '{chain_name}', skipping...")
            continue
        
        chain_definition = chain_definitions[chain_name]
        
        for interval, records in intervals_dict.items():
            if len(records) == 0:
                continue
            
            # Split into batches
            num_batches = (len(records) + RECORDS_PER_BATCH - 1) // RECORDS_PER_BATCH
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * RECORDS_PER_BATCH
                end_idx = min(start_idx + RECORDS_PER_BATCH, len(records))
                batch_records = records[start_idx:end_idx]
                
                job = BatchJob(
                    chain_name=chain_name,
                    chain_definition=chain_definition,
                    interval=interval,
                    batch_records=batch_records,
                    batch_idx=batch_idx,
                    interval_total=num_batches
                )
                all_jobs.append(job)
    
    print(f"Total batch jobs prepared: {len(all_jobs)}")
    print()
    
    return all_jobs


# ============================================================================
# Incremental Save Functions
# ============================================================================

class ResultsCache:
    """Thread-safe cache for accumulating results before saving"""
    def __init__(self):
        self.cache = {}  # {chain_name: {interval: {batch_idx: results}}}
        self.lock = threading.Lock()
        self.completed_intervals = set()  # Track completed intervals
        
    def add_batch_result(self, chain_name: str, interval: str, batch_idx: int, 
                        results: List[Dict], interval_total: int):
        """Add a batch result to cache"""
        with self.lock:
            if chain_name not in self.cache:
                self.cache[chain_name] = {}
            if interval not in self.cache[chain_name]:
                self.cache[chain_name][interval] = {}
            
            self.cache[chain_name][interval][batch_idx] = results
            
            # Check if interval is complete
            if len(self.cache[chain_name][interval]) == interval_total:
                self.completed_intervals.add((chain_name, interval))
    
    def get_completed_intervals(self) -> List[Tuple[str, str]]:
        """Get and clear list of completed intervals"""
        with self.lock:
            completed = list(self.completed_intervals)
            self.completed_intervals.clear()
            return completed
    
    def get_interval_results(self, chain_name: str, interval: str) -> List[Dict]:
        """Get sorted results for a completed interval"""
        with self.lock:
            if chain_name not in self.cache or interval not in self.cache[chain_name]:
                return []
            
            batch_dict = self.cache[chain_name][interval]
            # Sort by batch index and flatten
            sorted_batches = sorted(batch_dict.items(), key=lambda x: x[0])
            flattened = []
            for _, results in sorted_batches:
                flattened.extend(results)
            
            return flattened
    
    def remove_interval(self, chain_name: str, interval: str):
        """Remove an interval from cache after saving"""
        with self.lock:
            if chain_name in self.cache and interval in self.cache[chain_name]:
                del self.cache[chain_name][interval]


def load_existing_results(output_path: Path) -> Dict:
    """
    Load existing results if the output file exists.
    
    Args:
        output_path: Path to output JSON file
    
    Returns:
        Existing results dictionary or empty dict
    """
    if output_path.exists():
        return load_json(output_path)
    else:
        return {}


def save_completed_intervals(output_path: Path, results_cache: ResultsCache):
    """
    Save all completed intervals from cache to file.
    
    Args:
        output_path: Path to output JSON file
        results_cache: ResultsCache object
    """
    completed = results_cache.get_completed_intervals()
    
    if not completed:
        return
    
    with file_lock:
        # Load existing results
        all_results = load_existing_results(output_path)
        
        # Add completed intervals
        for chain_name, interval in completed:
            interval_results = results_cache.get_interval_results(chain_name, interval)
            
            if chain_name not in all_results:
                all_results[chain_name] = {}
            
            all_results[chain_name][interval] = interval_results
            
            # Remove from cache
            results_cache.remove_interval(chain_name, interval)
            
            print(f"  ✅ Saved: {chain_name} - {interval} ({len(interval_results)} records)")
        
        # Save back to file
        save_json(output_path, all_results)


# ============================================================================
# Main Processing Loop
# ============================================================================

def process_all_chains(
    similarity_samples: Dict[str, Dict[str, List]],
    chain_definitions: Dict[str, str],
    output_path: Path,
    save_interval: int = 100
) -> None:
    """
    Main processing loop: process all batches concurrently with full parallelization.
    
    Args:
        similarity_samples: Dictionary of similarity samples
        chain_definitions: Dictionary of chain definitions
        output_path: Path to save output
        save_interval: Save results after every N completed batches (default: 100)
    """
    print("=" * 70)
    print("Starting LLM-based threshold determination")
    print("=" * 70)
    print()
    
    # Prepare all batch jobs
    all_jobs = prepare_all_batch_jobs(similarity_samples, chain_definitions)
    
    if len(all_jobs) == 0:
        print("⚠ No jobs to process!")
        return
    
    # Initialize results cache
    results_cache = ResultsCache()
    
    # Statistics
    total_jobs = len(all_jobs)
    completed_count = 0
    failed_jobs = []
    
    print("=" * 70)
    print(f"Processing {total_jobs} batches with {MAX_CONCURRENT} concurrent workers")
    print(f"Will save results every {save_interval} completed batches")
    print("=" * 70)
    print()
    
    # Process all jobs with maximum concurrency
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        # Submit all jobs
        future_to_job = {executor.submit(process_single_batch, job): job for job in all_jobs}
        
        # Process results as they complete
        with tqdm(total=total_jobs, desc="Overall Progress", unit="batch") as pbar:
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                
                try:
                    chain_name, interval, batch_idx, results = future.result()
                    
                    # Add to cache
                    results_cache.add_batch_result(
                        chain_name, interval, batch_idx, results, job.interval_total
                    )
                    
                    completed_count += 1
                    pbar.update(1)
                    
                    # Periodic save
                    if completed_count % save_interval == 0:
                        save_completed_intervals(output_path, results_cache)
                    
                except Exception as e:
                    failed_jobs.append((job, str(e)))
                    print(f"\n❌ Failed: {job} - Error: {str(e)}")
                    pbar.update(1)
    
    # Final save of remaining completed intervals
    print("\n" + "=" * 70)
    print("Saving remaining results...")
    print("=" * 70)
    save_completed_intervals(output_path, results_cache)
    
    # Summary
    print("\n" + "=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print(f"Total batches: {total_jobs}")
    print(f"Completed: {completed_count - len(failed_jobs)}")
    print(f"Failed: {len(failed_jobs)}")
    print(f"Results saved to: {output_path}")
    
    if failed_jobs:
        print("\n⚠ Failed jobs:")
        for job, error in failed_jobs[:10]:  # Show first 10
            print(f"  - {job}: {error[:100]}")
        if len(failed_jobs) > 10:
            print(f"  ... and {len(failed_jobs) - 10} more")
    else:
        print("\n✅ All batches completed successfully!")
    
    print("=" * 70)


# ============================================================================
# Command-line Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='LLM-based threshold calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--records_per_batch',
        type=int,
        default=RECORDS_PER_BATCH,
        help=f'Number of records per LLM request (default: {RECORDS_PER_BATCH})'
    )
    
    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=MAX_CONCURRENT,
        help=f'Maximum concurrent requests (default: {MAX_CONCURRENT})'
    )
    
    parser.add_argument(
        '--retry_wait',
        type=int,
        default=RETRY_WAIT,
        help=f'Wait time (seconds) before retry on failure (default: {RETRY_WAIT})'
    )
    
    parser.add_argument(
        '--save_interval',
        type=int,
        default=LLM_SAVE_INTERVAL,
        help=f'Save results after every N completed batches (default: {LLM_SAVE_INTERVAL})'
    )
    
    parser.add_argument(
        '--api_url',
        type=str,
        default=API_URL,
        help=f'LLM API URL (default: {API_URL})'
    )
    
    parser.add_argument(
        '--api_key',
        type=str,
        default=API_KEY,
        help=f'API key for authentication (default: {API_KEY})'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Update global configuration from arguments
    global RECORDS_PER_BATCH, MAX_CONCURRENT, RETRY_WAIT, API_URL, API_KEY
    RECORDS_PER_BATCH = args.records_per_batch
    MAX_CONCURRENT = args.max_concurrent
    RETRY_WAIT = args.retry_wait
    API_URL = args.api_url
    API_KEY = args.api_key
    
    print("\n" + "=" * 70)
    print("LLM-based Threshold Calibration")
    print("=" * 70)
    print(f"Records per batch: {RECORDS_PER_BATCH}")
    print(f"Max concurrent requests: {MAX_CONCURRENT}")
    print(f"Retry wait time: {RETRY_WAIT}s")
    print(f"Save interval: {args.save_interval} batches")
    print(f"API URL: {API_URL}")
    print()
    
    # Load data
    similarity_samples = load_similarity_samples(SIMILARITY_SAMPLES_PATH)
    chain_definitions = load_chain_definitions(CHAIN_DEFINITIONS_PATH)
    
    # Process all chains
    process_all_chains(similarity_samples, chain_definitions, OUTPUT_PATH, args.save_interval)


if __name__ == "__main__":
    main()

