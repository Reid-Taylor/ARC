import os 
import glob 
from json import JSONDecoder


data_dir = "data/ARC-AGI/data/training/"
json_files = glob.glob(os.path.join(data_dir, "*.json"))
all_data = {}
for file_path in json_files:
    with open(file_path, 'r') as f:
        data = JSONDecoder().decode(f.read())
        all_data[os.path.basename(file_path)] = data

print(all_data.get('ff805c23.json', 'File not found'))