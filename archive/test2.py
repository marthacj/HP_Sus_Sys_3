import pandas as pd
from collections import OrderedDict
import math


# Sample data (assuming you start with a pandas DataFrame and convert to a dictionary)
# df = pd.read_csv('your_csv_file.csv')
# machine_usage_data = df.to_dict()

# Example input dictionary for demonstration
machine_usage_data = {
    'Unnamed: 0': {0: 'Machine 1', 1: 'Machine 2'},
    '#Cores': {0: 24.0, 1: 28.0},
    'Core Highest min': {0: 0.173611, 1: 0.555556},
    'Core Highest max': {0: 67.708333, 1: 78.263889},
    'Core Highest avg': {0: 0.815864, 1: 0.714057}
}

# Clean and sort the input data
cleaned_machine_usage_data = {}
for mud_k, mud_v in machine_usage_data.items():
    cleaned_machine_usage_data[mud_k] = {k: v for k, v in mud_v.items() if not (isinstance(v, float) and math.isnan(v))}
    cleaned_machine_usage_data[mud_k] = OrderedDict(sorted(cleaned_machine_usage_data[mud_k].items()))

# Rename and modify 'Unnamed: 0'
cleaned_machine_usage_data['machine-id-list'] = cleaned_machine_usage_data.pop('Unnamed: 0')
cleaned_machine_usage_data['machine-id-list'] = {k: v[8:] for k, v in cleaned_machine_usage_data['machine-id-list'].items()}  # Adjust slicing as needed

# Round numerical values to 6 decimal places
for mud_k, mud_v in cleaned_machine_usage_data.items():
    for k, v in mud_v.items():
        if isinstance(v, float):
            cleaned_machine_usage_data[mud_k][k] = round(v, 6)

# Remove empty sub-dictionaries
cleaned_machine_usage_data = {k: v for k, v in cleaned_machine_usage_data.items() if v}

# Find the common indices across all sub-dictionaries
common_indices = set.intersection(*(set(sub_dict.keys()) for sub_dict in cleaned_machine_usage_data.values()))

# Flip the dictionary structure
flipped_data = {}

# Iterate only over the common indices
for idx in common_indices:
    flipped_data[idx] = {category: cleaned_machine_usage_data[category][idx] for category in cleaned_machine_usage_data}

# Replace 'avg' with 'average usage' in keys
flipped_data = {idx: {category.replace('avg', 'average usage'): value for category, value in data.items()} for idx, data in flipped_data.items()}

# Output the new structure before converting to a string
for idx, data in flipped_data.items():
    print(f"{idx}: {data}")

# Convert to string and replace \n with a space
machine_usage_data_str = str(flipped_data).replace(r'\n', ' ')

# Output the final string (or use it as needed in your application)
print(machine_usage_data_str)