import os
import pathlib
from pathlib import Path

# Path to the data directory
# Sort the files in the data directory by time

data_dir = os.path.join(os.getcwd(), "data")
data_dir = Path(data_dir)
for file in sorted(data_dir.iterdir(), key=os.path.getmtime):
    print(file)
