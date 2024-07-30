import pandas as pd
import yaml
import sys
from datetime import datetime, timezone


# Define the input and output file paths
original_CSV_filepath = r'data\CSV1038-0610-0614-day.csv'
modified_CSV_filepath = r'data\modified_CSV1038-0610-0614-day.csv'
manifest_filepath = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\manifest1\NEW_z2_G4_Sci.yaml'


def process_row(row, start_date, duration):
    return {
        'timestamp': start_date,
        'device/emissions-embodied': row['device/emissions-embodied'],
        'cpu/thermal-design-power': row['cpu/thermal-design-power'],
        'vcpus-total': row['cores'],
        'vcpus-allocated': row['cores'],
        'cpu/utilization': float(row['CPU_average']),
        'max-cpu-wattage': row['max-cpu-wattage'],
        'gpu/utilization': float(row['GPU_average']),
        'max-gpu-wattage': row['max-gpu-wattage'],
        'total-MB-sent': float(row['Total_MB_Sent']),
        'total-MB-received': float(row['Total_MB_Received']),
        'instance-type': row['machine-family'],
        'machine-code': row['Host Name'],
        'time-reserved': int(row['time-reserved']),
        'grid/carbon-intensity': int(row['grid/carbon-intensity']),
        'device/expected-lifespan': int(row['device/expected-lifespan']),
        'duration': int(duration),
        'network-intensity': float(row['network-intensity']),
        'machine': int(1)
    }

def process_csv(original_CSV_filepath, modified_CSV_filepath):
    # Read the CSV file
    with open(original_CSV_filepath, 'r') as file:
        lines = file.readlines()
    
    # Extract the 'duration' value from the metadata
    duration = None
    for line in lines:
        if "Total Secs:" in line:
            parts = line.split(',')
            for i, part in enumerate(parts):
                if "Total Secs:" in part:
                    duration = parts[i + 1].strip()
                    
            # if duration is not None:
            #     break
        if "Data Pulled:" in line:
            parts = line.split(',')
            for i, part in enumerate(parts):
                if "Data Pulled:" in part:
                    data_pulled = parts[i].strip()
                    date_range = data_pulled.replace("Data Pulled: ", "")
                    start_date_str, end_date_str = date_range.split(" - ")

                    # Define the date format
                    date_format = "%b %d %Y %H:%M:%S"

                    # Parse the date strings into datetime objects
                    start_date = datetime.strptime(start_date_str, date_format)
                    start_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                    end_date = datetime.strptime(end_date_str, date_format)

                    # Print the datetime objects
                    print("Start Date:", start_date)
                    print("End Date:", end_date)
                   
                    break
    
    if duration is None:
        raise ValueError("Duration value not found in the file")
    if start_date is None:
        raise ValueError("Start date value not found in the file")
    if end_date is None:
        raise ValueError("End date value not found in the file")
    
    # Find the start of the data table
    start_row = 0
    for i, line in enumerate(lines):
        if 'Host Name' in line:
            start_row = i
            break
    
        # Load the CSV data including the headers rows
    df = pd.read_csv(original_CSV_filepath, header=None, skiprows=start_row)

    # Extract the first row as headers for the first column
    first_column_header = df.iloc[0, 0]

    # Extract the second row as headers for columns 2 onwards
    remaining_headers = df.iloc[1]

    # Combine headers
    headers = [first_column_header] + remaining_headers[1:].tolist()

    # Assign the combined headers to the DataFrame
    df.columns = headers

    # Drop the first two rows which were used for headers
    df = df.drop([0, 1]).reset_index(drop=True)

    # Optionally: Rename specific columns if needed
    replace_dict = {
        '#Cores': 'cores',
        'CPU\nHighest\navg': 'CPU_average',
        'GPU\navg': 'GPU_average',
        'Total MB\nSent': 'Total_MB_Sent',
        'Total MB\nReceived': 'Total_MB_Received'
    }

    # Replace column names using the replace_dict
    df.rename(columns=replace_dict, inplace=True)

    print(df)
    
    df['machine-family'] = ''
    df['max-cpu-wattage'] = ''
    df['max-gpu-wattage'] = ''
    df['cpu/thermal-design-power'] = ''
    df['device/emissions-embodied'] = ''
    df['time-reserved'] = '157788000'
    df['grid/carbon-intensity'] = 31
    df['device/expected-lifespan'] = 157788000
    df['time-reserved'] = 157788000
    df['network-intensity'] = 0.000124


    # Iterate through the DataFrame and update the 'machine-family' column based on 'cores'
    for index, row in df.iterrows():
        if row['cores'] == '24':
            df.at[index, 'cores'] = 24
            df.at[index, 'machine-family'] = 'z2 mini'
            df.at[index, 'max-cpu-wattage'] = 280
            df.at[index, 'max-gpu-wattage'] = 70
            df.at[index, 'cpu/thermal-design-power'] = 90
            df.at[index, 'device/emissions-embodied'] = 370.14
        elif row['cores'] == '28':
            df.at[index, 'cores'] = 28
            df.at[index, 'machine-family'] = 'Z4R G4'
            df.at[index, 'max-cpu-wattage'] = 1400
            df.at[index, 'max-gpu-wattage'] = 230
            df.at[index, 'cpu/thermal-design-power'] = 165
            df.at[index, 'device/emissions-embodied'] = 306
        if row['GPU_average'] == '0':
            df.at[index, 'GPU_average'] = 1
            
         

    print(df)

    print(df.columns)
    print(df['GPU_average'])
    templates = []

    # Iterate through each row in the DataFrame and process it
    for _, row in df.iterrows():
        if pd.isna(row['Host Name']) or row['Host Name'] == '':
            continue
        template = process_row(row, start_date, duration)
        templates.append(template)

    print(templates)
    
    # Output the modified DataFrame to a new CSV file
    df.to_csv(modified_CSV_filepath, index=False)
    
    return modified_CSV_filepath, int(duration), start_date, end_date, templates


modified_csv_path, duration, start_date, end_date, templates = process_csv(original_CSV_filepath, modified_CSV_filepath)

def generate_manifest(manifest_filepath, modified_CSV_filepath, duration, templates):
    # Define the manifest structure
    manifest = {
        'name': 'sci-calculation',
        'description': """Calculate operational carbon from CPU utilization, GPU utilization, and network usage.
            SCI is ISO-recognized standard for reporting carbon costs of running software, takes into account all the energy used by the application; below includes CPU energy and network energy.""",
        'initialize': {
            'outputs': ['yaml'],
            'plugins': {
                # 'tdp-finder': {
                #     'method': 'CSVLookup',
                #     'path': 'builtin',
                #     'global-config': {
                #         'filepath': modified_CSV_filepath,
                #         'query': {
                #             'host': 'instance-type'
                #         },
                #         'output': [['cores', 'num-cores'], ['CPU_average', 'cpu/utilization'], ['GPU_average','gpu/utilization'], ['Total_MB_Sent', 'total-MB-sent'], ['Total_MB_Received', 'total-MB-received'], ['machine-family', 'machine-family']]
                #     }
                # },
                'group-by': {
                    'path': 'builtin',
                    'method': 'GroupBy',
                },
                'interpolate': {
                  'method': 'Interpolation',
                  'path': 'builtin',
                  'global-config': {
                      'method': 'linear',
                      'x':  [0, 10, 50, 100] ,
                      'y':  [0.12, 0.32, 0.75, 1.02] ,
                    'input-parameter': "cpu/utilization",
                    'output-parameter': "cpu-factor" }
                },
                'cpu-factor-to-wattage': { # Determines power drawn by CPU at exact utilisation % by multiplying scaling factor and TDP
                  'method': 'Multiply',
                  'path': 'builtin',
                  'global-config': {
                    'input-parameters':  ["cpu-factor", "cpu/thermal-design-power"] ,
                    'output-parameter': "cpu-wattage"}
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
                'gpu-utilisation-to-wattage': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["gpu-utilization", "max-gpu-wattage"],
                        'output-parameter': "gpu-wattage"
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
                'cpu-utilisation-to-wattage': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["cpu-utilization", "cpu/thermal-design-power"],
                        'output-parameter': "cpu-wattage"
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
                        'output': "cpu-energy-raw"
                    }
                },
                'calculate-vcpu-ratio': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "vcpus-allocated",
                        'denominator': "vcpus-total",
                        'output': "vcpu-ratio"
                    }
                },
                'correct-cpu-energy-for-vcpu-ratio': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "cpu-energy-raw",
                        'denominator': "vcpu-ratio",
                        'output': "cpu/energy"
                    }
                },
                'energy-sent': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["total-MB-sent", "network-intensity"],
                        'output-parameter': "energy-sent-joules"
                    }
                },
                'energy-received': {
                    'method': 'Multiply',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["total-MB-received", "network-intensity"],
                        'output-parameter': "energy-received-joules"
                    }
                },
                'sum-network-energy-joules': {
                    'method': 'Sum',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["energy-sent-joules", "energy-received-joules"],
                        'output-parameter': "total-energy-network-joules"
                    }
                },
                'total-network-energy-to-kwh': {
                    'method': 'Divide',
                    'path': 'builtin',
                    'global-config': {
                        'numerator': "total-energy-network-joules",
                        'denominator': 3600000,
                        'output': "network/energy"
                    }
                },
                'sum-energy-components': {
                    'method': 'Sum',
                    'path': 'builtin',
                    'global-config': {
                        'input-parameters': ["cpu/energy", "gpu/energy", "network/energy"],
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
                        'input-parameters': ["energy", "grid/carbon-intensity"],
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
                }
            }
        },
        'tree': {
            'children': {
                'child': {
                    'pipeline': [
        'group-by',
        # 'interpolate',
        # 'cpu-factor-to-wattage',
        'gpu-utilisation-percentage-to-decimal',
        'gpu-utilisation-to-wattage',
        'gpu-wattage-times-duration',
        'gpu-wattage-to-energy-kwh',
        'cpu-utilisation-percentage-to-decimal',
        'cpu-utilisation-to-wattage',
        'cpu-wattage-times-duration',
        'cpu-wattage-to-energy-kwh',
        'calculate-vcpu-ratio',
        'correct-cpu-energy-for-vcpu-ratio',
        'energy-sent',
        'energy-received',
        'sum-network-energy-joules',
        'total-network-energy-to-kwh',
        'sum-energy-components',
        'sci-embodied',
        'operational-carbon',
        'sum-carbon',
        'sci'
    ],
    'config': {
        'group-by': {
            'group': ['machine-code']
    }},

                    'inputs': templates
                }},
            }
        }
    
    print(manifest)
    # Save the manifest to a YAML file
    with open(manifest_filepath, 'w', encoding='utf-8') as file:
        yaml.dump(manifest, file, default_flow_style=False, sort_keys=False)


# Generate the manifest file with the extracted duration value
generate_manifest(manifest_filepath, modified_csv_path, duration, templates)

print(f"CSV file has been modified and saved as {modified_CSV_filepath}")
print(f"Manifest file has been created with the extracted duration value at {manifest_filepath}")
print(f"Extracted duration value: {duration}")
