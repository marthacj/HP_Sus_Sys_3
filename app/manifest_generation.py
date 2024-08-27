import pandas as pd
import yaml
from datetime import datetime, timezone
import csv
from dateutil.parser import parse
import os

def convert_xlsx_to_csv(excel_file):
    df = pd.read_excel(excel_file, sheet_name='WS-Data')

    # Define the output CSV file path
    csv_file = r'data/uploaded_file.csv'

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

    return csv_file


def process_row(row, start_date, duration):
    return {
        'timestamp': start_date,
        'device/emissions-embodied': row['device/emissions-embodied'],
        'cpu/thermal-design-power': row['cpu/thermal-design-power'],
        'gpu/thermal-design-power': row['gpu/thermal-design-power'],
        'vcpus-total': row['cores'],
        'vcpus-allocated': row['cores'],
        'cpu/utilization': float(row['CPU_average']),
        'max-machine-wattage': row['max-machine-wattage'],
        'gpu/utilization': float(row['GPU_average']),
        # 'max-gpu-wattage': row['max-gpu-wattage'],
        'total-MB-sent': float(row['Total_MB_Sent']),
        'total-MB-received': float(row['Total_MB_Received']),
        'instance-type': row['machine-family'],
        'machine-code': row['Host Name'],
        'time-reserved': int(row['time-reserved']),
        'grid/carbon-intensity': int(row['grid/carbon-intensity']),
        'device/expected-lifespan': int(row['device/expected-lifespan']),
        'duration': int(duration),
        'network-intensity': float(row['network-intensity']),
        'machine': int(1),
        'memory/thermal-design-power': row['memory/thermal-design-power'],
        'cpu-memory/utilization': float(row['CPU_memory_average']),
        'gpu-memory/utilization': float(row['GPU_memory_average']),
        'PUE': float(row['PUE'])

    }

def process_csv(original_CSV_filepath, modified_CSV_filepath):
    with open(original_CSV_filepath, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)

    duration = None
    start_date = None
    end_date = None
    analysis_window = None

    for line in lines:
        if line and 'Data Pulled:' in line[0]:
            date_range = line[0].split(':', 1)[1].strip()
            start_date_str, end_date_str = date_range.split('-')
            start_date = parse(start_date_str.strip()).strftime('%Y-%m-%dT%H:%M:%S.000Z')
            end_date = parse(end_date_str.strip()).strftime('%Y-%m-%dT%H:%M:%S.000Z')
            
            # Find 'Total Secs:' in the same line
            try:
                total_secs_index = line.index('Total Secs:')
                duration = line[total_secs_index + 1].strip()
                
            except ValueError:
                print("Warning: 'Total Secs:' not found in the expected line")
        
        elif line and 'Analysis Window:' in line[0]:
            analysis_window = line[0].split(':', 1)[1].strip()                   
    
            analysis_window = line[0].split(':', 1)[1].strip()                   
    

    # Find the start of the data table
    start_row = 0
    for i, line in enumerate(lines):
        if 'Host Name' in line:
            start_row = i
            break
        # Load the CSV data including the headers rows
    df = pd.read_csv(original_CSV_filepath, header=None, skiprows=start_row)
    first_column_header = df.iloc[0, 0]

    # Extract the second row as headers for columns 2 onwards
    remaining_headers = df.iloc[1]
    headers = [first_column_header] + remaining_headers[1:].tolist()
    df.columns = headers
    
    # Drop the first two rows which were used for headers
    df = df.drop([0, 1]).reset_index(drop=True)
    # print(df)
    df.drop(df.tail(3).index, inplace=True)
    # print(df)
    # Optionally: Rename specific columns if needed
    replace_dict = {
        '#Cores': 'cores',
        'CPU\nHighest\navg': 'CPU_average',
        'GPU\navg': 'GPU_average',
        'Total MB\nSent': 'Total_MB_Sent',
        'Total MB\nReceived': 'Total_MB_Received',
        'avg': 'CPU_memory_average',
        'MEM\navg': 'GPU_memory_average'
    }

    required_columns = list(replace_dict.keys()) + ['Host Name']

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in headers]

    if missing_columns != []:
        raise ValueError(f"Some or all of the following required columns are missing in the uploaded Excel file: {', '.join(missing_columns)}")
    

    # Replace column names using the replace_dict
    df.rename(columns=replace_dict, inplace=True)
    # print("Columns after renaming:", df.columns)

    df['machine-family'] = ''
    df['max-machine-wattage'] = ''
    df['cpu/thermal-design-power'] = ''
    df['gpu/thermal-design-power'] = ''
    df['device/emissions-embodied'] = ''
    df['time-reserved'] = '157788000'
    df['grid/carbon-intensity'] = 133
    df['device/expected-lifespan'] = 157784760
    df['time-reserved'] = 157784760
    df['network-intensity'] = 0.000124
    df['memory/thermal-design-power'] = ''
    df['PUE'] = 1.4
    missing_values = []
    if duration is None:
        missing_values.append("Duration")
    if start_date is None:
        missing_values.append("Start Date")
    if missing_values:
        raise ValueError(f"The following values were not found in the file: {', '.join(missing_values)}")

    # Iterate through the DataFrame and update the 'machine-family' column based on 'cores'
    for index, row in df.iterrows():
        if row['cores'] == '24':
            df.at[index, 'cores'] = 24
            df.at[index, 'machine-family'] = 'z2 mini'
            df.at[index, 'max-machine-wattage'] = 280
            df.at[index, 'cpu/thermal-design-power'] = 90
            df.at[index, 'gpu/thermal-design-power'] = 70
            df.at[index, 'device/emissions-embodied'] = 370.14 * 1000
            df.at[index, 'memory/thermal-design-power'] = 17
        elif row['cores'] == '28':
            df.at[index, 'cores'] = 28
            df.at[index, 'machine-family'] = 'Z4R G4'
            df.at[index, 'max-machine-wattage'] = 1400
            df.at[index, 'gpu/thermal-design-power'] = 230
            df.at[index, 'cpu/thermal-design-power'] = 165
            df.at[index, 'device/emissions-embodied'] = 306 * 1000
            df.at[index, 'memory/thermal-design-power'] = 48
        if row['GPU_average'] == '0':
            df.at[index, 'GPU_average'] = 0.1
    # print(df['CPU_average'])

    # print(df.columns)
    templates = []

    # Iterate through each row in the DataFrame and process it
    for _, row in df.iterrows():
        if pd.isna(row['Host Name']) or row['Host Name'] == '':
            continue
        template = process_row(row, start_date, duration)
        templates.append(template)
    # print(templates)
    
    # Output the modified DataFrame to a new CSV file
    df.to_csv(modified_CSV_filepath, index=False)
    # print(templates)
    return modified_CSV_filepath, int(duration), start_date, end_date, templates, analysis_window
    

def generate_manifest(manifest_filepath, modified_CSV_filepath, duration, templates):
    # Define the manifest structure
    manifest = {
        'name': 'sci-calculation',
        'description': """Calculate operational carbon from CPU utilization, GPU utilization, and network usage.
            SCI is ISO-recognized standard for reporting carbon costs of running software, takes into account all the energy used by the application; below includes CPU energy and network energy.""",
        'initialize': {
            'outputs': ['yaml'],
            'plugins': {
                'interpolate-cpu': {
                    'method': 'Interpolation',
                    'path': 'builtin',
                    'global-config': {
                        'method': 'linear',
                        'x': [0, 10, 50, 100],
                        'y': [0.12, 0.32, 0.75, 1.02],
                        'input-parameter': "cpu-utilization",
                        'output-parameter': "cpu-factor"
                    }
                },
                'interpolate-gpu': {
                    'method': 'Interpolation',
                    'path': 'builtin',
                    'global-config': {
                        'method': 'linear',
                        'x': [0, 10, 50, 100],
                        'y': [0.15, 0.32, 0.75, 0.99],
                        'input-parameter': "gpu-utilization",
                        'output-parameter': "gpu-factor"
                    }
                },
                'cpu-factor-to-wattage': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["cpu-factor", "cpu/thermal-design-power"],
                        'output-parameter': "cpu-wattage"
                    }
                },
                'gpu-factor-to-wattage': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["gpu-factor", "gpu/thermal-design-power"],
                        'output-parameter': "gpu-wattage"
                    }
                },
                'gpu-utilisation-percentage-to-decimal': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "gpu/utilization",
                        'denominator': 100,
                        'output': "gpu-utilization"
                    }
                },
                'gpu-wattage-times-duration': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["gpu-wattage", "duration"],
                        'output-parameter': "gpu-wattage-times-duration"
                    }
                },
                'gpu-wattage-to-energy-kwh': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "gpu-wattage-times-duration",
                        'denominator': 3600000,
                        'output': "gpu/energy"
                    }
                },
                'cpu-utilisation-percentage-to-decimal': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "cpu/utilization",
                        'denominator': 100,
                        'output': "cpu-utilization"
                    }
                },
                'cpu-wattage-times-duration': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["cpu-wattage", "duration"],
                        'output-parameter': "cpu-wattage-times-duration"
                    }
                },
                'cpu-wattage-to-energy-kwh': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "cpu-wattage-times-duration",
                        'denominator': 3600000,
                        'output': "cpu/energy"
                    }
                },
                'cpu-memory-utilisation-percentage-to-decimal': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "cpu-memory/utilization",
                        'denominator': 100,
                        'output': "cpu-memory-utilization"
                    }
                },
                'gpu-memory-utilisation-percentage-to-decimal': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "gpu-memory/utilization",
                        'denominator': 100,
                        'output': "gpu-memory-utilization"
                    }
                },
                'add-cpu-gpu-memory-utilization': {
                    'method': 'Sum',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["cpu-memory-utilization", "gpu-memory-utilization"],
                        'output-parameter': "cpu-gpu-combined-memory-utilization"
                    }
                },
                'average-cpu-gpu-memory-utilization': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "cpu-gpu-combined-memory-utilization",
                        'denominator': 2,
                        'output': "cpu-gpu-average-memory-utilization"
                    }
                },
                'cpu-gpu-average-memory-utilization-to-wattage': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["cpu-gpu-average-memory-utilization", "memory/thermal-design-power"],
                        'output-parameter': "memory-wattage"
                    }
                },
                'memory-wattage-times-duration': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["memory-wattage", "duration"],
                        'output-parameter': "memory-wattage-times-duration"
                    }
                },
                'memory-wattage-to-energy-kwh': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "memory-wattage-times-duration",
                        'denominator': 3600000,
                        'output': "memory/energy"
                    }
                },
                'sum-energy-components': {
                    'method': 'Sum',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["cpu/energy", "gpu/energy", "memory/energy"],
                        'output-parameter': "energy"
                    }
                },
                'sci-embodied': {
                    'method': 'SciEmbodied',
                    'path': 'builtin'
                },
                'operational-carbon': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["pue-energy", "grid/carbon-intensity"],
                        'output-parameter': "carbon-operational"
                    }
                },
                'sum-carbon': {
                    'method': 'Sum',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["carbon-operational", "carbon-embodied"],
                        'output-parameter': "carbon"
                    }
                },
                'sci': {
                    'method': 'Sci',
                    'path': 'builtin',
                    'global-config': {
                        'functional-unit': 'machine'
                    }
                },
                'pue-times-energy': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ['energy', 'PUE'],
                        'output-parameter': 'pue-energy'}}
            }
        },
        'tree': {
            'children': {
                'child': {
                    'pipeline': {
                        'regroup': ['machine-code'],
                        'compute': [
                            'cpu-utilisation-percentage-to-decimal',
                            'interpolate-cpu',
                            'cpu-factor-to-wattage',
                            'gpu-utilisation-percentage-to-decimal',
                            'interpolate-gpu',
                            'gpu-factor-to-wattage',
                            'gpu-wattage-times-duration',
                            'gpu-wattage-to-energy-kwh',
                            'cpu-wattage-times-duration',
                            'cpu-wattage-to-energy-kwh',
                            'cpu-memory-utilisation-percentage-to-decimal',
                            'gpu-memory-utilisation-percentage-to-decimal',
                            'add-cpu-gpu-memory-utilization',
                            'average-cpu-gpu-memory-utilization',
                            'cpu-gpu-average-memory-utilization-to-wattage',
                            'memory-wattage-times-duration',
                            'memory-wattage-to-energy-kwh',
                            'sum-energy-components',
                            'pue-times-energy',
                            'sci-embodied',
                            'operational-carbon',
                            'sum-carbon',
                            'sci'
                        ]
                    },
                    'inputs': templates
                }
            }
        }
    }

    # Write the manifest to a file
    with open(manifest_filepath, 'w', encoding='utf-8') as file:
        yaml.dump(manifest, file, default_flow_style=False, sort_keys=False)


def safe_generate_manifest(manifest_filepath, modified_csv_path, duration, templates):
    try:
        generate_manifest(manifest_filepath, modified_csv_path, duration, templates)
        return True
    except Exception as e:
        print(f"Error generating manifest: {str(e)}")
        return False

def safe_print_file_info(filepath, description):
    if os.path.exists(filepath):
        print(f"\n{description} has been created at {filepath}")
    else:
        print(f"Warning: {description} was not found at {filepath}")

if __name__ == '__main__':
    excel_file = r'data/1038-0610-0614-day.xlsx'
    #excel_file = r'data/IF-sanity-check.xlsx'
    csv_file = convert_xlsx_to_csv(excel_file)
    # Define the input and output file paths
    original_CSV_filepath = csv_file
    modified_CSV_filepath = r'data/modified_CSV1038-0610-0614-day.csv'
    manifest_filepath = r'manifest1/NEW_z2_G4_Sci.yaml'

    modified_csv_path, duration, start_date, end_date, templates, analysis_window = process_csv(original_CSV_filepath, modified_CSV_filepath)
    # Generate the manifest file with the extracted duration value
    safe_generate_manifest(manifest_filepath, modified_csv_path, duration, templates)

    print(f"CSV file has been modified and saved as {modified_CSV_filepath}")
    print(f"Manifest file has been created with the extracted duration value at {manifest_filepath}")
    print(f"Extracted duration value: {duration}\n\n\n")
