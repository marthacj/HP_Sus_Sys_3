import pandas as pd
import yaml
import sys
from datetime import datetime


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
    
    # Load the CSV data into a DataFrame, skipping the metadata rows
    df = pd.read_csv(original_CSV_filepath, skiprows=start_row)
    
    # Rename the 'Host Name' column to 'host'
    df.rename(columns={'Host Name': 'host'}, inplace=True)
    replace_dict = {
    '#Cores': 'cores',
    'CPU\nHighest\navg': 'CPU_average',
    'GPU\navg': 'GPU_average',
    'Total MB\nSent': 'Total_MB_Sent',
    'Total MB\nReceived': 'Total_MB_Received'
}

    df.replace(replace_dict, inplace=True)
    print(df)
    
    df['machine-family'] = ''

    # Iterate through the DataFrame and update the 'machine-family' column based on 'cores'
    for index, row in df.iterrows():
        if row['cores'] == 24:
            df.at[index, 'machine-family'] = 'z2 mini'
        elif row['cores'] == 28:
            df.at[index, 'machine-family'] = 'G4 Z4R'
    print(df)
    sys.exit()

    
    # Remove the first 4 rows
    # df = df.iloc[4:].reset_index(drop=True)
    
    # Output the modified DataFrame to a new CSV file
    df.to_csv(modified_CSV_filepath, index=False)
    
    return modified_CSV_filepath, int(duration), start_date, end_date


# Define the input and output file paths
original_CSV_filepath = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\data\CSV1038-0610-0614-day.csv'
modified_CSV_filepath = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\data\modified_CSV1038-0610-0614-day.csv'
manifest_filepath = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\manifest1\NEW_z2_G4_Sci.yaml'


modified_csv_path, duration, start_date, end_date = process_csv(original_CSV_filepath, modified_CSV_filepath)

def generate_manifest(manifest_filepath, modified_CSV_filepath, duration):
    # Define the manifest structure
    manifest = {
        'name': 'sci-calculation',
        'description': """Calculate operational carbon from CPU utilization using the Teads curve and then get operational and embodied carbon too. 
            SCI is ISO-recognized standard for reporting carbon costs of running software, takes into account all the energy used by the application; below includes CPU energy and network energy.""",
        'initialize': {
            'outputs': ['yaml'],
            'plugins': {
                'tdp-finder': {
                    'method': 'CSVLookup',
                    'path': 'builtin',
                    'global-config': {
                        'filepath': modified_CSV_filepath,
                        'query': {
                            'host': 'instance-type'
                        },
                        'output': [['cores', 'num-cores'], ['CPU_average', 'cpu/utilization'], ['GPU_average','gpu/utilization'], ['Total_MB_Sent', 'total-MB-sent'], ['Total_MB_Received', 'total-MB-received']]
                    }
                },
                'group-by': {
                    'path': 'builtin',
                    'method': 'GroupBy',
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
        'tdp-finder',
        'group-by',
        # - interpolate
        # - cpu-factor-to-wattage
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
            'group': ['instance-type', 'cores']
    }},
    'defaults': [
    {'time-reserved': 157788000},  # need to check. the length of time the hardware is reserved for use by the software: BIG IMPACT ON RESULTS
    {'grid/carbon-intensity': 31},  # this is the number for Equinix DC 2023, 35 is the general number for London (June 2024)
    {'device/expected-lifespan': 157788000},  # 5 years in seconds == the length of time, in seconds, between a component's manufacture and its disposal
    {'resources-reserved': 'vcpus-allocated'},
    {'resources-total': 'vcpus-total'},
    {'machine': 1},  # this is for 1 machine right now as have taken the average for machines rather than data for all machines: BIG IMPACT ON RESULTS
    {'duration': duration},
    {'network-intensity': 0.000124},  # kWh/MB
    {'timestamp': start_date}
    ]
},
                    'inputs': [
                        {'instance-type': 'ld71r18u44dws'},
                        # {'instance-type': 'ld71r16u15ws'},
                        # {'instance-type': 'ld71r18u44fws'},
                        # {'instance-type': 'ld71r16u13ws'},
                        # {'instance-type': 'ld71r18u44bws'},
                        # {'instance-type': 'ld71r18u44cws'},
                        # {'instance-type': 'ld71r16u14ws'},
                        # {'instance-type': 'ld71r18u44ews'},
                        
                    ]
                }
            }
        }
    

    # Save the manifest to a YAML file
    with open(manifest_filepath, 'w') as file:
        yaml.dump(manifest, file, default_flow_style=False)




# # Output the modified DataFrame to a new CSV file
# df.to_csv(modified_CSV_filepath, index=False)

# Generate the manifest file with the extracted duration value
generate_manifest(manifest_filepath, modified_csv_path, duration)

print(f"CSV file has been modified and saved as {modified_CSV_filepath}")
print(f"Manifest file has been created with the extracted duration value at {manifest_filepath}")
print(f"Extracted duration value: {duration}")