"""
python convert.py --input output/resnet18/preds.npy --output submissions/ --model "resnet18_v1"

"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

def convert_predictions(input_path, output_path=None, submission_name=None):
    """
    Convert model predictions from .npy format to CSV format suitable for submission.
    
    Args:
        input_path (str): Path to the .npy file containing predictions
        output_path (str, optional): Path to save the CSV file or directory
        submission_name (str, optional): Name to use in the output filename
    
    Returns:
        str: Path to the saved CSV file
    """
    print(f"Loading predictions from {input_path}")
    
    data = np.load(input_path)
    
    # Convert logits to class predictions (argmax along the class dimension)
    data = np.argmax(data, axis=1)
    
    # Create DataFrame with predictions
    df = pd.DataFrame(data)
    
    df.index += 1
    
    # Generate filename and handle directory paths
    if submission_name is None:
        # Try to extract model name from the input path
        submission_name = Path(input_path).parent.name
    
    filename = f"submission_{submission_name}.csv"
    
    # Handle output path
    if output_path is None:
        # Use current directory if no path specified
        final_path = filename
    elif os.path.isdir(output_path):
        # If output_path is a directory, join with the filename
        os.makedirs(output_path, exist_ok=True)  # Ensure directory exists
        final_path = os.path.join(output_path, filename)
    else:
        # Use output_path as provided (assuming it's a full filepath)
        final_path = output_path
    
    # Save to CSV with proper headers
    df.to_csv(final_path, header=['Category'], index_label='Id')
    print(f"Saved submission file to {final_path}")
    
    # Print sample of predictions for verification
    print("\nSample predictions (first 5 entries):")
    print(df.head())
    
    return final_path

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert model predictions to CSV format")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the .npy file containing predictions")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to save the CSV file (default: submission_<model_name>.csv)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model name to use in the output filename")
    
    args = parser.parse_args()
    
    # Convert predictions
    convert_predictions(args.input, args.output, args.model)
