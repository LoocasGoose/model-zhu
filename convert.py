"""
Utility to convert model predictions to Kaggle submission format

Usage:
    python convert.py --input output/resnet18/preds.npy --output submissions/ --model "resnet18_v1"
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Optional, Union, Tuple


def convert_predictions(
    input_path: Union[str, Path], 
    output_path: Optional[Union[str, Path]] = None, 
    submission_name: Optional[str] = None,
    expected_rows: int = 75929
) -> Tuple[str, pd.DataFrame]:
    """
    Convert model predictions from .npy format to CSV format suitable for submission.
    
    Args:
        input_path: Path to the .npy file containing predictions
        output_path: Path to save the CSV file or directory
        submission_name: Name to use in the output filename
        expected_rows: Expected number of rows in the final submission (default: 75929)
    
    Returns:
        Tuple of (path to the saved CSV file, DataFrame with predictions)
    
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the predictions don't match the expected shape or format
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading predictions from {input_path}")
    
    try:
        data = np.load(input_path)
    except Exception as e:
        raise ValueError(f"Failed to load predictions: {str(e)}")
    
    # Validate data format
    if len(data.shape) < 2:
        raise ValueError(f"Predictions must be 2D with shape (n_samples, n_classes), got {data.shape}")
    
    # Convert logits to class predictions (argmax along the class dimension)
    data = np.argmax(data, axis=1)
    
    # Validate number of predictions
    if len(data) != expected_rows:
        print(f"Warning: Found {len(data)} predictions, but expected {expected_rows}")
    
    # Create DataFrame with predictions
    df = pd.DataFrame(data, columns=["Category"])
    
    # Start indexing from 1 for submission format
    df.index += 1
    
    # Generate filename and handle directory paths
    if submission_name is None:
        # Try to extract model name from the input path
        submission_name = input_path.parent.name
    
    filename = f"submission_{submission_name}.csv"
    
    # Handle output path
    if output_path is None:
        # Use current directory if no path specified
        final_path = Path(filename)
    else:
        output_path = Path(output_path)
        if output_path.is_dir():
            # If output_path is a directory, join with the filename
            output_path.mkdir(exist_ok=True, parents=True)  # Ensure directory exists
            final_path = output_path / filename
        else:
            # Use output_path as provided (assuming it's a full filepath)
            final_path = output_path
            final_path.parent.mkdir(exist_ok=True, parents=True)  # Ensure parent directory exists
    
    # Save to CSV with proper headers
    df.to_csv(final_path, header=['Category'], index_label='Id')
    print(f"Saved submission file to {final_path}")
    
    # Print sample of predictions for verification
    print("\nSample predictions (first 5 entries):")
    print(df.head())
    
    # Print statistics about predictions
    print(f"\nPrediction statistics:")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique classes: {df['Category'].nunique()}")
    print(f"  Class range: {df['Category'].min()} to {df['Category'].max()}")
    
    return str(final_path), df


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert model predictions to CSV format")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the .npy file containing predictions")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to save the CSV file (default: submission_<model_name>.csv)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model name to use in the output filename")
    parser.add_argument("--rows", "-r", type=int, default=75929,
                        help="Expected number of rows in the final submission (default: 75929)")
    
    args = parser.parse_args()
    
    # Convert predictions
    convert_predictions(args.input, args.output, args.model, args.rows)
