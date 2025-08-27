#!/usr/bin/env python3
"""
Multi-run accuracy calculator for HDC models.
This module provides a function to run a Python script multiple times and compute the average accuracy.
"""

import subprocess
import re
import statistics
from pathlib import Path
from typing import List, Optional

def run_script_multiple_times(script_path: str, num_runs: int = 30, verbose: bool = False) -> float:
    """
    Run a Python script multiple times and return the average accuracy.
    
    Args:
        script_path (str): Path to the Python script to run
        num_runs (int): Number of times to run the script (default: 30)
        verbose (bool): Whether to print detailed output for each run
    
    Returns:
        float: Average accuracy across all runs
    
    Raises:
        FileNotFoundError: If the script path does not exist
        ValueError: If no accuracy value is found in the output
    """
    # Validate script path
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    accuracies = []
    accuracy_pattern = re.compile(r"Validation Accuracy: (\d+\.\d+)%")
    
    for run in range(1, num_runs + 1):
        if verbose:
            print(f"Run {run}/{num_runs}...")
        
        # Run the script and capture output
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            cwd=script_path.parent  # Run in the script's directory
        )
        
        # Check for errors
        if result.returncode != 0:
            print(f"Warning: Run {run} failed with return code {result.returncode}")
            print(f"Stderr: {result.stderr}")
            continue
        
        # Parse accuracy from output
        match = accuracy_pattern.search(result.stdout)
        if match:
            accuracy = float(match.group(1))
            accuracies.append(accuracy)
            if verbose:
                print(f"Run {run} accuracy: {accuracy:.2f}%")
        else:
            print(f"Warning: Could not find accuracy in output for run {run}")
            # Try alternative patterns if the first one doesn't match
            alt_patterns = [
                r"accuracy: (\d+\.\d+)%",
                r"Accuracy: (\d+\.\d+)%",
                r"acc: (\d+\.\d+)%",
            ]
            for pattern in alt_patterns:
                alt_match = re.search(pattern, result.stdout, re.IGNORECASE)
                if alt_match:
                    accuracy = float(alt_match.group(1))
                    accuracies.append(accuracy)
                    if verbose:
                        print(f"Run {run} accuracy: {accuracy:.2f}%")
                    break
            else:
                print(f"Could not parse accuracy from output. Full output:\n{result.stdout}")
    
    if not accuracies:
        raise ValueError("No accuracy values were captured from any run")
    
    # Calculate average accuracy
    avg_accuracy = statistics.mean(accuracies)
    
    if verbose:
        print(f"\nCompleted {len(accuracies)} successful runs")
        print(f"Individual accuracies: {[f'{acc:.2f}%' for acc in accuracies]}")
        print(f"Average accuracy: {avg_accuracy:.2f}%")
        print(f"Standard deviation: {statistics.stdev(accuracies):.2f}%")
    
    return avg_accuracy

def main():
    """Command-line interface for multi-run accuracy calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a Python script multiple times and calculate average accuracy")
    parser.add_argument("script", help="Path to the Python script to run")
    parser.add_argument("-n", "--num-runs", type=int, default=30, help="Number of runs (default: 30)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        avg_accuracy = run_script_multiple_times(args.script, args.num_runs, args.verbose)
        print(f"\nFinal average accuracy: {avg_accuracy:.2f}%")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
