#!/usr/bin/env python3
import time
from tqdm import tqdm

# Test tqdm progress bar
print("Testing tqdm progress bar...")
for i in tqdm(range(10), desc="Test Progress"):
    time.sleep(0.5)  # Simulate work

print("Done!") 