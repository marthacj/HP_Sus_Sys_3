import pandas as pd
from typing import List, Dict
import openai  
import sys
from LLM import *
from manifest_generation import *
from ollama import Client 


excel_file = r'data/OFFICIAL-TEST-SET-1.xlsx'

def convert_test_xlsx_to_csv(excel_file):
    df = pd.read_excel(excel_file, sheet_name='WS-Data')

    # Define the output CSV file path
    csv_file = r'data\TEST_uploaded_file.csv'

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

    return csv_file

csv_file = convert_test_xlsx_to_csv(excel_file)

original_CSV_filepath = csv_file
modified_CSV_filepath = r'data\TESTmodified_CSV.csv'


# Process the CSV file and extract the duration value, start date, end date, and templates to create the manifest file
modified_csv_path, duration, start_date, end_date, templates, analysis_window = process_csv(original_CSV_filepath, modified_CSV_filepath)

num_of_machines = '8'
merged_df = pd.read_csv(r'embeddings\merged_df.csv')

sentences = []
add_context_to_sentences(sentences, duration, start_date, end_date, analysis_window, num_of_machines=num_of_machines)


def get_cpu_vals_per_machine(merged_df):
    cpu_util_column = 'central processing unit average utilisation percent'
    machine_column = 'Machine'

    machine_and_cpu = []
    # Iterate through the DataFrame rows
    for index, row in merged_df.iterrows():
        machine = row[machine_column]
        cpu_util = row[cpu_util_column]
        cpu_util = round(cpu_util, 2)
        machine_and_cpu.append((machine, ':', cpu_util))

    return machine_and_cpu


def set_up_test_dataframe_values(merged_df):
    u44dws_emissions = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44dws', 'carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS'].values[0]
    total_emissions = merged_df['carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS'].sum()
    total_emissions = round(total_emissions, 2)
    max_gpu_util_row = merged_df.loc[merged_df['graphics processing unit average utilisation percent'].idxmax()]
    # print('max_gpu_util_row:', max_gpu_util_row)
    machine_with_max_gpu = max_gpu_util_row['Machine']
    max_gpu_util = max_gpu_util_row['graphics processing unit average utilisation percent']
    max_gpu_util = round(max_gpu_util, 2)
    machine_and_cpu = get_cpu_vals_per_machine(merged_df)
    cpu_utilization_str = "Here\'s a summary of the central processing unit usage for all the machines:\n"
    cpu_utilization_str += "central processing unit average utilisation percent:\n"
    for machine, _, cpu in machine_and_cpu:
        cpu_utilization_str += f"{machine}: {cpu:.2f}%\n"   
    average_cpu_util = merged_df['central processing unit average utilisation percent'].mean()
    # GEt the machine and the value for the highest average CPU utilisation percent
    machine_with_max_cpu = merged_df.loc[merged_df['central processing unit average utilisation percent'].idxmax()]
    max_cpu_util = machine_with_max_cpu['central processing unit average utilisation percent']
    max_cpu_util = round(max_cpu_util, 2)
    # get machine with the lowest average CPU utilisation percent
    machine_with_min_cpu = merged_df.loc[merged_df['central processing unit average utilisation percent'].idxmin()]
    min_cpu_util = machine_with_min_cpu['central processing unit average utilisation percent']
    min_cpu_util = round(min_cpu_util, 2)
    # get the machine with the highest central processing unit maximum utilisation percent
    machine_with_max_cpu_max = merged_df.loc[merged_df['central processing unit maximum utilisation percent'].idxmax()]
    max_cpu_util_max = machine_with_max_cpu_max['central processing unit maximum utilisation percent']
    max_cpu_util_max = round(max_cpu_util_max, 2)
    # get the machine with the lowest central processing unit maximum utilisation percent
    machine_with_min_cpu_max = merged_df.loc[merged_df['central processing unit maximum utilisation percent'].idxmin()]
    min_cpu_util_max = machine_with_min_cpu_max['central processing unit maximum utilisation percent']
    min_cpu_util_max = round(min_cpu_util_max, 2)
    # get the machine where central processing unit total seconds over 80% is highest
    max_cpu_seconds_over_80_idx = merged_df['central processing unit total seconds over 80%'].idxmax()
    max_cpu_seconds_over_80 = merged_df.loc[max_cpu_seconds_over_80_idx, 'central processing unit total seconds over 80%']
    machine_with_max_cpu_seconds_over_80 = merged_df.loc[max_cpu_seconds_over_80_idx, 'Machine']
    machine_with_max_cpu_seconds_over_80_avg_cpu = merged_df.loc[max_cpu_seconds_over_80_idx, 'central processing unit average utilisation percent']
    max_emissions_idx = merged_df['carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS'].idxmax()
    machine_with_max_emissions = merged_df.loc[max_emissions_idx, 'Machine']
    max_emissions = merged_df.loc[max_emissions_idx, 'carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS']
    max_emissions = round(max_emissions, 2)

    return u44dws_emissions, total_emissions, machine_with_max_gpu, max_gpu_util, cpu_utilization_str, average_cpu_util, machine_with_max_cpu, max_cpu_util, machine_with_min_cpu, min_cpu_util, machine_with_max_cpu_max, max_cpu_util_max, machine_with_min_cpu_max, min_cpu_util_max, machine_with_max_cpu_seconds_over_80, max_cpu_seconds_over_80, machine_with_max_cpu_seconds_over_80_avg_cpu, machine_with_max_emissions, max_emissions

u44dws_emissions, total_emissions, machine_with_max_gpu, max_gpu_util, cpu_utilization_str, average_cpu_util, machine_with_max_cpu, max_cpu_util, machine_with_min_cpu, min_cpu_util, machine_with_max_cpu_max, max_cpu_util_max, machine_with_min_cpu_max, min_cpu_util_max,  machine_with_max_cpu_seconds_over_80, max_cpu_seconds_over_80, machine_with_max_cpu_seconds_over_80_avg_cpu, machine_with_max_emissions, max_emissions = set_up_test_dataframe_values(merged_df)

def get_carbon_emission_vals_per_machine(merged_df):
    carbon_emission_column = 'carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS'
    r16u13ws_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u13ws', carbon_emission_column].values[0]
    r16u15ws_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u15ws', carbon_emission_column].values[0]
    r16u14ws_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u14ws', carbon_emission_column].values[0]
    r18u44ews_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44ews', carbon_emission_column].values[0]
    r18u44cws_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44cws', carbon_emission_column].values[0]
    r18u44fws_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44fws', carbon_emission_column].values[0]
    r18u44dws_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44dws', carbon_emission_column].values[0]
    r18u44bws_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44bws', carbon_emission_column].values[0]

    return r16u13ws_emimssion, r16u15ws_emimssion, r16u14ws_emimssion, r18u44ews_emimssion, r18u44cws_emimssion, r18u44fws_emimssion, r18u44dws_emimssion, r18u44bws_emimssion


r16u13ws_emimssion, r16u15ws_emimssion, r16u14ws_emimssion, r18u44ews_emimssion, r18u44cws_emimssion, r18u44fws_emimssion, r18u44dws_emimssion, r18u44bws_emimssion = get_carbon_emission_vals_per_machine(merged_df)

def get_embodied_emission_vals_per_machine(merged_df):
    carbon_emission_column = 'embodied carbon (gCO2eq)'
    r16u13ws_embodied_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u13ws', carbon_emission_column].values[0]
    r16u15ws_embodied_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u15ws', carbon_emission_column].values[0]
    r16u14ws_embodied_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u14ws', carbon_emission_column].values[0]
    r18u44ews_embodied_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44ews', carbon_emission_column].values[0]
    r18u44cws_embodied_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44cws', carbon_emission_column].values[0]
    r18u44fws_embodied_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44fws', carbon_emission_column].values[0]
    r18u44dws_embodied_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44dws', carbon_emission_column].values[0]
    r18u44bws_embodied_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44bws', carbon_emission_column].values[0]

    return r16u13ws_embodied_emimssion, r16u15ws_embodied_emimssion, r16u14ws_embodied_emimssion, r18u44ews_embodied_emimssion, r18u44cws_embodied_emimssion, r18u44fws_embodied_emimssion, r18u44dws_embodied_emimssion, r18u44bws_embodied_emimssion

r16u13ws_embodied_emimssion, r16u15ws_embodied_emimssion, r16u14ws_embodied_emimssion, r18u44ews_embodied_emimssion, r18u44cws_embodied_emimssion, r18u44fws_embodied_emimssion, r18u44dws_embodied_emimssion, r18u44bws_embodied_emimssion = get_embodied_emission_vals_per_machine(merged_df)

def get_operational_carbon_vals_per_machine(merged_df):
    carbon_emission_column = 'operational carbon (gCO2eq)'
    r16u13ws_operational_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u13ws', carbon_emission_column].values[0]
    r16u15ws_operational_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u15ws', carbon_emission_column].values[0]
    r16u14ws_operational_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u14ws', carbon_emission_column].values[0]
    r18u44ews_operational_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44ews', carbon_emission_column].values[0]
    r18u44cws_operational_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44cws', carbon_emission_column].values[0]
    r18u44fws_operational_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44fws', carbon_emission_column].values[0]
    r18u44dws_operational_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44dws', carbon_emission_column].values[0]
    r18u44bws_operational_emimssion = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44bws', carbon_emission_column].values[0]

    return r16u13ws_operational_emimssion, r16u15ws_operational_emimssion, r16u14ws_operational_emimssion, r18u44ews_operational_emimssion, r18u44cws_operational_emimssion, r18u44fws_operational_emimssion, r18u44dws_operational_emimssion, r18u44bws_operational_emimssion

r16u13ws_operational_emimssion, r16u15ws_operational_emimssion, r16u14ws_operational_emimssion, r18u44ews_operational_emimssion, r18u44cws_operational_emimssion, r18u44fws_operational_emimssion, r18u44dws_operational_emimssion, r18u44bws_operational_emimssion = get_operational_carbon_vals_per_machine(merged_df)

def get_cpu_average_utilisation_vals_per_machine(merged_df):
    CPU_col = 'central processing unit average utilisation percent'
    r16u13ws_average_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u13ws', CPU_col].values[0]
    r16u15ws_average_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u15ws', CPU_col].values[0]
    r16u14ws_average_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u14ws', CPU_col].values[0]
    r18u44ews_average_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44ews', CPU_col].values[0]
    r18u44cws_average_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44cws', CPU_col].values[0]
    r18u44fws_average_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44fws', CPU_col].values[0]
    r18u44dws_average_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44dws', CPU_col].values[0]
    r18u44bws_average_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44bws', CPU_col].values[0]

    return r16u13ws_average_cpu, r16u15ws_average_cpu, r16u14ws_average_cpu, r18u44ews_average_cpu, r18u44cws_average_cpu, r18u44fws_average_cpu, r18u44dws_average_cpu, r18u44bws_average_cpu

r16u13ws_average_cpu, r16u15ws_average_cpu, r16u14ws_average_cpu, r18u44ews_average_cpu, r18u44cws_average_cpu, r18u44fws_average_cpu, r18u44dws_average_cpu, r18u44bws_average_cpu = get_cpu_average_utilisation_vals_per_machine(merged_df)

def get_cpu_maximum_utilisation_vals_per_machine(merged_df):
    CPU_col = 'central processing unit maximum utilisation percent'
    r16u13ws_maximum_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u13ws', CPU_col].values[0]
    r16u15ws_maximum_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u15ws', CPU_col].values[0]
    r16u14ws_maximum_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u14ws', CPU_col].values[0]
    r18u44ews_maximum_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44ews', CPU_col].values[0]
    r18u44cws_maximum_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44cws', CPU_col].values[0]
    r18u44fws_maximum_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44fws', CPU_col].values[0]
    r18u44dws_maximum_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44dws', CPU_col].values[0]
    r18u44bws_maximum_cpu = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44bws', CPU_col].values[0]

    return r16u13ws_maximum_cpu, r16u15ws_maximum_cpu, r16u14ws_maximum_cpu, r18u44ews_maximum_cpu, r18u44cws_maximum_cpu, r18u44fws_maximum_cpu, r18u44dws_maximum_cpu, r18u44bws_maximum_cpu

r16u13ws_maximum_cpu, r16u15ws_maximum_cpu, r16u14ws_maximum_cpu, r18u44ews_maximum_cpu, r18u44cws_maximum_cpu, r18u44fws_maximum_cpu, r18u44dws_maximum_cpu, r18u44bws_maximum_cpu = get_cpu_maximum_utilisation_vals_per_machine(merged_df)

def get_gpu_average_utilisation_vals_per_machine(merged_df):
    GPU_col = 'graphics processing unit average utilisation percent'
    r16u13ws_gpu_average = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u13ws', GPU_col].values[0]
    r16u15ws_gpu_average = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u15ws', GPU_col].values[0]
    r16u14ws_gpu_average = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u14ws', GPU_col].values[0]
    r18u44ews_gpu_average = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44ews', GPU_col].values[0]
    r18u44cws_gpu_average = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44cws', GPU_col].values[0]
    r18u44fws_gpu_average = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44fws', GPU_col].values[0]
    r18u44dws_gpu_average = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44dws', GPU_col].values[0]
    r18u44bws_gpu_average = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44bws', GPU_col].values[0]

    return r16u13ws_gpu_average, r16u15ws_gpu_average, r16u14ws_gpu_average, r18u44ews_gpu_average, r18u44cws_gpu_average, r18u44fws_gpu_average, r18u44dws_gpu_average, r18u44bws_gpu_average

r16u13ws_gpu_average, r16u15ws_gpu_average, r16u14ws_gpu_average, r18u44ews_gpu_average, r18u44cws_gpu_average, r18u44fws_gpu_average, r18u44dws_gpu_average, r18u44bws_gpu_average = get_gpu_average_utilisation_vals_per_machine(merged_df)

def get_gpu_average_utilisation_memory_vals_per_machine(merged_df):
    GPU_col = 'graphics processing unit average memory utilisation percent'
    r16u13ws_gpu_average_memory = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u13ws', GPU_col].values[0]
    r16u15ws_gpu_average_memory = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u15ws', GPU_col].values[0]
    r16u14ws_gpu_average_memory = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r16u14ws', GPU_col].values[0]
    r18u44ews_gpu_average_memory = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44ews', GPU_col].values[0]
    r18u44cws_gpu_average_memory = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44cws', GPU_col].values[0]
    r18u44fws_gpu_average_memory = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44fws', GPU_col].values[0]
    r18u44dws_gpu_average_memory = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44dws', GPU_col].values[0]
    r18u44bws_gpu_average_memory = merged_df.loc[merged_df['Machine'].str.strip() == 'ld71r18u44bws', GPU_col].values[0]

    return r16u13ws_gpu_average_memory, r16u15ws_gpu_average_memory, r16u14ws_gpu_average_memory, r18u44ews_gpu_average_memory, r18u44cws_gpu_average_memory, r18u44fws_gpu_average_memory, r18u44dws_gpu_average_memory, r18u44bws_gpu_average_memory

r16u13ws_gpu_average_memory, r16u15ws_gpu_average_memory, r16u14ws_gpu_average_memory, r18u44ews_gpu_average_memory, r18u44cws_gpu_average_memory, r18u44fws_gpu_average_memory, r18u44dws_gpu_average_memory, r18u44bws_gpu_average_memory = get_gpu_average_utilisation_memory_vals_per_machine(merged_df)

def question_0_knowledge_provided():
   question_0_provided_knowledge = f"""
    Here is your context for a question I will ask you:
ld71r16u13ws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r16u13ws_emimssion}
ld71r16u15ws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r16u15ws_emimssion}
ld71r18u44fws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r18u44fws_emimssion}
ld71r16u14ws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r16u14ws_emimssion}
ld71r18u44dws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r18u44dws_emimssion}
ld71r18u44bws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r18u44bws_emimssion}
ld71r18u44cws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r18u44cws_emimssion}
ld71r18u44ews carbon emissions (gco2eq) - use this for questions about carbon emissions = {r18u44ews_emimssion}
ld71r16u15ws embodied carbon (gco2eq) = {r16u15ws_embodied_emimssion}
ld71r16u14ws embodied carbon (gco2eq) = {r16u14ws_embodied_emimssion}
ld71r16u13ws embodied carbon (gco2eq) = {r16u13ws_embodied_emimssion}
ld71r18u44ews embodied carbon (gco2eq) ={r18u44ews_embodied_emimssion}
ld71r18u44cws embodied carbon (gco2eq) = {r18u44cws_embodied_emimssion}
ld71r18u44fws embodied carbon (gco2eq) = {r18u44fws_embodied_emimssion}
ld71r18u44dws embodied carbon (gco2eq) = {r18u44dws_embodied_emimssion}
ld71r18u44bws embodied carbon (gco2eq) = {r18u44bws_embodied_emimssion}
ld71r18u44bws operational carbon (gco2eq) = {r18u44bws_operational_emimssion}
ld71r16u13ws operational carbon (gco2eq) = {r16u13ws_operational_emimssion}
ld71r16u14ws operational carbon (gco2eq) = {r16u14ws_operational_emimssion}
ld71r16u15ws operational carbon (gco2eq) = {r16u15ws_operational_emimssion}
ld71r18u44ews operational carbon (gco2eq) = {r18u44ews_operational_emimssion}
ld71r18u44fws operational carbon (gco2eq) = {r18u44fws_operational_emimssion}
ld71r18u44cws operational carbon (gco2eq) = {r18u44cws_operational_emimssion}
ld71r18u44dws operational carbon (gco2eq) = {r18u44dws_operational_emimssion}
{sentences[0]}
{sentences[11]}
ld71r18u44fws central processing unit average utilisation percent = {r18u44fws_average_cpu}
ld71r16u13ws central processing unit average utilisation percent = {r16u13ws_average_cpu}
ld71r18u44ews central processing unit maximum utilisation percent = {r18u44ews_maximum_cpu}
ld71r16u14ws central processing unit average utilisation percent = {r16u14ws_average_cpu}
ld71r18u44ews central processing unit average utilisation percent = {r18u44ews_average_cpu}
ld71r16u15ws central processing unit average utilisation percent = {r16u15ws_average_cpu}
ld71r16u14ws central processing unit maximum utilisation percent = {r16u14ws_maximum_cpu}
ld71r18u44dws central processing unit average utilisation percent = {r18u44dws_average_cpu}
ld71r16u15ws central processing unit maximum utilisation percent = {r16u15ws_maximum_cpu}
ld71r18u44dws central processing unit maximum utilisation percent = {r18u44dws_maximum_cpu}
ld71r16u13ws central processing unit maximum utilisation percent = {r16u13ws_maximum_cpu}
ld71r18u44fws central processing unit maximum utilisation percent = {r18u44fws_maximum_cpu}
ld71r18u44cws central processing unit average utilisation percent = {r18u44cws_average_cpu}
ld71r18u44cws central processing unit maximum utilisation percent = {r18u44cws_maximum_cpu}
ld71r18u44bws central processing unit average utilisation percent = {r18u44bws_average_cpu}
ld71r18u44bws central processing unit maximum utilisation percent = {r18u44bws_maximum_cpu}
ld71r16u14ws central processing unit total seconds over 80percent  = 0
ld71r16u13ws central processing unit total seconds over 80percent  = 96000
ld71r16u15ws central processing unit total seconds over 80percent  = 0
ld71r18u44bws central processing unit total seconds over 80percent  = 0
ld71r18u44ews core maximum utilisation percent (single core of highest usage) = 75.656
ld71r16u15ws core maximum utilisation percent (single core of highest usage) = 63.04
ld71r16u13ws core maximum utilisation percent (single core of highest usage) = 99.872
ld71r16u13ws megabytes sent across network traffic = 859.692
ld71r16u14ws core maximum utilisation percent (single core of highest usage) = 53.594
ld71r16u13ws number of cores = 28
ld71r18u44dws megabytes sent across network traffic = 832.185
ld71r18u44cws number of cores = 24
ld71r16u14ws number of cores = 28"""
   return question_0_provided_knowledge

question_0_provided_knowledge = question_0_knowledge_provided()
# Define the number of runs
num_runs = 10


def question_1_knowledge_provided():
    question_1_provided_knowledge = f"""
    Here is your context for a question I will ask you:
ld71r16u13ws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r16u13ws_emimssion}
ld71r16u14ws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r16u14ws_emimssion}
ld71r18u44fws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r18u44fws_emimssion}
ld71r18u44cws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r18u44cws_emimssion}
{sentences[11]}
ld71r18u44bws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r18u44bws_emimssion}
ld71r16u15ws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r16u15ws_emimssion}
ld71r18u44dws carbon emissions (gco2eq) - use this for questions about carbon emissions = {r18u44dws_emimssion}
{sentences[0]}
ld71r18u44ews carbon emissions (gco2eq) - use this for questions about carbon emissions = {r18u44ews_emimssion}
ld71r18u44cws embodied carbon (gco2eq) = {r18u44cws_embodied_emimssion}
ld71r18u44fws embodied carbon (gco2eq) = {r18u44fws_embodied_emimssion}
ld71r18u44bws embodied carbon (gco2eq) = {r18u44bws_embodied_emimssion}
ld71r18u44dws embodied carbon (gco2eq) = {r18u44dws_embodied_emimssion}
ld71r16u15ws embodied carbon (gco2eq) = {r16u15ws_embodied_emimssion}
ld71r16u14ws embodied carbon (gco2eq) = {r16u14ws_embodied_emimssion}
ld71r16u13ws embodied carbon (gco2eq) = {r16u13ws_embodied_emimssion}
ld71r18u44ews embodied carbon (gco2eq) = {r18u44ews_embodied_emimssion}
ld71r18u44bws operational carbon (gco2eq) = {r18u44bws_operational_emimssion}
{sentences[10]}
ld71r16u13ws operational carbon (gco2eq) = {r16u13ws_operational_emimssion}
ld71r18u44fws operational carbon (gco2eq) = {r18u44fws_operational_emimssion}
ld71r16u14ws operational carbon (gco2eq) = {r16u14ws_operational_emimssion}
ld71r16u15ws operational carbon (gco2eq) = {r16u15ws_operational_emimssion}
ld71r18u44cws operational carbon (gco2eq) = {r18u44cws_operational_emimssion}
ld71r18u44fws central processing unit average utilisation percent = {r18u44fws_average_cpu}
ld71r18u44ews operational carbon (gco2eq) = {r18u44ews_operational_emimssion}
ld71r18u44dws operational carbon (gco2eq) = {r18u44dws_operational_emimssion}
ld71r18u44cws central processing unit average utilisation percent = {r18u44cws_average_cpu}
ld71r18u44bws central processing unit average utilisation percent = {r18u44bws_average_cpu}
ld71r16u13ws central processing unit average utilisation percent = {r16u13ws_average_cpu}
ld71r16u14ws central processing unit average utilisation percent = {r16u14ws_average_cpu}
ld71r18u44dws central processing unit average utilisation percent = {r18u44dws_average_cpu}
ld71r18u44ews central processing unit average utilisation percent = {r18u44ews_average_cpu}
ld71r16u15ws central processing unit average utilisation percent = {r16u15ws_average_cpu}
ld71r18u44fws central processing unit maximum utilisation percent = {r18u44fws_maximum_cpu}
ld71r16u15ws central processing unit maximum utilisation percent = {r16u15ws_maximum_cpu}
{sentences[5]}
ld71r18u44dws central processing unit maximum utilisation percent = {r18u44dws_maximum_cpu}
ld71r18u44ews central processing unit maximum utilisation percent = {r18u44ews_maximum_cpu}
ld71r18u44bws central processing unit maximum utilisation percent = {r18u44bws_maximum_cpu}
ld71r16u14ws central processing unit maximum utilisation percent = {r16u14ws_maximum_cpu}
ld71r18u44cws central processing unit maximum utilisation percent = {r18u44cws_maximum_cpu}
ld71r16u13ws central processing unit maximum utilisation percent = {r16u13ws_maximum_cpu}
ld71r16u13ws core average utilisation percent (single core of highest usage) = 90.46
{sentences[4]}
ld71r18u44cws graphics processing unit average utilisation percent = 48.01
ld71r16u15ws core average utilisation percent (single core of highest usage) = 18.1
ld71r18u44dws graphics processing unit average utilisation percent = 60.99
ld71r18u44bws core average utilisation percent (single core of highest usage) = 34.79
ld71r16u14ws core average utilisation percent (single core of highest usage) = 24.68
ld71r18u44fws graphics processing unit average utilisation percent = 54.12
ld71r18u44fws core average utilisation percent (single core of highest usage) = 83.92
ld71r18u44ews core average utilisation percent (single core of highest usage) = 61.39
ld71r16u13ws graphics processing unit average utilisation percent = 39.03
"""
    return question_1_provided_knowledge

question_1_provided_knowledge = question_1_knowledge_provided()

def question_2_knowledge_provided():
    question_2_provided_knowledge = f"""
     Here is your context for a question I will ask you:
ld71r18u44bws graphics processing unit average utilisation percent = {r18u44bws_gpu_average}
ld71r18u44cws graphics processing unit average utilisation percent = {r18u44cws_gpu_average}
ld71r18u44fws graphics processing unit average utilisation percent = {r18u44fws_gpu_average}
ld71r18u44ews graphics processing unit average utilisation percent = {r18u44ews_gpu_average}
ld71r18u44dws graphics processing unit average utilisation percent = {r18u44dws_gpu_average}
ld71r16u13ws graphics processing unit average utilisation percent = {r16u13ws_gpu_average}
ld71r16u15ws graphics processing unit average utilisation percent = {r16u15ws_gpu_average}
ld71r16u14ws graphics processing unit average utilisation percent = {r16u14ws_gpu_average}
ld71r18u44cws graphics processing unit average memory utilisation percent = {r18u44cws_gpu_average_memory}
ld71r18u44bws graphics processing unit average memory utilisation percent =  {r18u44bws_gpu_average_memory}
If the graphics processing unit average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like a higher compute power with more cores. Around 75% utlilisation is very efficient as it is making the most of the resources (good for sustainability), yet work is not slowed down by compute limitations.
ld71r18u44fws graphics processing unit average memory utilisation percent = {r18u44fws_gpu_average_memory}
If the graphics processing unit maximum utilisation percent is above 95%, it indicates that the machine is close to performing at maximum compute power which means it may be causing a bottleneck. However, if the average is low, then by itself it is not a reason to change the compute power.
ld71r18u44dws graphics processing unit average memory utilisation percent = {r18u44dws_gpu_average_memory}
ld71r16u14ws graphics processing unit average memory utilisation percent = {r16u14ws_gpu_average_memory}
ld71r18u44ews graphics processing unit average memory utilisation percent = {r18u44ews_gpu_average_memory}
ld71r16u15ws graphics processing unit average memory utilisation percent = {r16u15ws_gpu_average_memory}
ld71r16u13ws graphics processing unit average memory utilisation percent = {r16u13ws_gpu_average_memory}
ld71r18u44bws graphics processing unit maximum utilisation percent = 95.03
ld71r18u44bws graphics processing unit maximum memory utilisation percent = 86
If the graphics processing unit average utilisation percent is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.
ld71r18u44fws graphics processing unit maximum utilisation percent = 46.9
ld71r18u44cws graphics processing unit maximum memory utilisation percent = 79
ld71r16u15ws graphics processing unit maximum memory utilisation percent = 67
ld71r18u44fws graphics processing unit maximum memory utilisation percent = 67
ld71r16u14ws graphics processing unit maximum memory utilisation percent = 36
ld71r16u13ws graphics processing unit maximum memory utilisation percent = 83
ld71r16u13ws graphics processing unit maximum utilisation percent = 89.31
ld71r16u15ws graphics processing unit maximum utilisation percent = 69.35
ld71r18u44dws graphics processing unit maximum memory utilisation percent = 63
ld71r18u44cws graphics processing unit maximum utilisation percent = 51.04
ld71r16u14ws graphics processing unit maximum utilisation percent = 83.97
ld71r18u44ews graphics processing unit maximum memory utilisation percent = 45
ld71r18u44dws graphics processing unit maximum utilisation percent = 90.25
ld71r18u44cws core average utilisation percent (single core of highest usage) = 55.98
If the central processing unit maximum utilisation percent is above 95%, it indicates that the machine is close to performing at maximum compute power which means it may be causing a bottleneck. However, if the average is low, then by itself it is not a reason to change the compute power.
ld71r18u44bws core average utilisation percent (single core of highest usage) = 34.79
ld71r18u44ews core average utilisation percent (single core of highest usage) = 61.39
ld71r18u44fws core average utilisation percent (single core of highest usage) = 83.92
ld71r18u44bws central processing unit average utilisation percent = 29.45
ld71r16u14ws core average utilisation percent (single core of highest usage) = 24.68
ld71r16u15ws core average utilisation percent (single core of highest usage) = 18.1
ld71r16u13ws central processing unit average utilisation percent = 89.13
ld71r16u15ws central processing unit average utilisation percent = 20.1
ld71r18u44dws core average utilisation percent (single core of highest usage) = 20.49
ld71r18u44fws central processing unit average utilisation percent = 84.37
ld71r18u44ews central processing unit average utilisation percent = 69.82
ld71r18u44cws central processing unit average utilisation percent = 57.09
ld71r16u14ws central processing unit average utilisation percent = 38.14
ld71r16u15ws core maximum utilisation percent (single core of highest usage) = 63.04
ld71r18u44ews graphics processing unit maximum utilisation percent = 75.76
ld71r16u15ws central processing unit maximum utilisation percent = 66.69
ld71r18u44dws central processing unit maximum utilisation percent = 71.96
ld71r18u44dws central processing unit average utilisation percent = 30.08
ld71r16u13ws central processing unit maximum utilisation percent = 97.51
    """
    return question_2_provided_knowledge

question_2_provided_knowledge = question_2_knowledge_provided()

def question_3_knowledge_provided():
    question_3_provided_knowledge = f"""
    Here is your context for a question I will ask you:
    ld71r18u44bws central processing unit average utilisation percent = 29.45
ld71r18u44fws central processing unit average utilisation percent = 84.37
ld71r18u44cws central processing unit average utilisation percent = 57.09
ld71r18u44dws central processing unit average utilisation percent = 30.08
ld71r18u44ews central processing unit average utilisation percent = 69.82
ld71r16u13ws central processing unit average utilisation percent = 89.13
ld71r16u15ws central processing unit average utilisation percent = 20.1
ld71r16u14ws central processing unit average utilisation percent = 38.14
If the central processing unit average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like being moved to a higher compute power with more cores.  Around 75% utlilisation is very efficient as it is making the most of the resources (good for sustainability), yet work is not slowed down by compute limitations.
If the central processing unit average utilisation percent is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.
ld71r18u44fws central processing unit maximum utilisation percent = 97.69
ld71r18u44dws central processing unit maximum utilisation percent = 71.96
ld71r18u44cws central processing unit maximum utilisation percent = 86.76
ld71r18u44bws central processing unit maximum utilisation percent = 64.72
ld71r16u14ws central processing unit maximum utilisation percent = 78.9
ld71r16u15ws central processing unit maximum utilisation percent = 66.69
ld71r16u13ws central processing unit maximum utilisation percent = 97.51
If the central processing unit maximum utilisation percent is above 95%, it indicates that the machine is close to performing at maximum compute power which means it may be causing a bottleneck. However, if the average is low, then by itself it is not a reason to change the compute power.  
ld71r18u44ews central processing unit maximum utilisation percent = 79.78
ld71r18u44bws central processing unit total seconds over 80percent  = 0
ld71r18u44cws core average utilisation percent (single core of highest usage) = 55.98
ld71r16u13ws central processing unit total seconds over 80percent  = 96000
ld71r18u44bws core average utilisation percent (single core of highest usage) = 34.79
ld71r16u14ws central processing unit total seconds over 80percent  = 0
ld71r18u44dws core average utilisation percent (single core of highest usage) = 20.49
ld71r18u44fws core average utilisation percent (single core of highest usage) = 83.92
ld71r16u15ws central processing unit total seconds over 80percent  = 0
ld71r16u15ws core average utilisation percent (single core of highest usage) = 18.1
ld71r16u14ws core average utilisation percent (single core of highest usage) = 24.68
ld71r18u44cws graphics processing unit average utilisation percent = 48.01
If the graphics processing unit average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like a higher compute power with more cores. Around 75% utlilisation is very efficient as it is making the most of the resources (good for sustainability), yet work is not slowed down by compute limitations.
ld71r18u44ews core average utilisation percent (single core of highest usage) = 61.39
ld71r16u13ws core average utilisation percent (single core of highest usage) = 90.46
If the graphics processing unit average utilisation percent is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.
ld71r18u44bws number of occurrences central processing unit went over 80percent  = 0
ld71r18u44dws graphics processing unit average utilisation percent = 60.99
ld71r18u44bws graphics processing unit average utilisation percent = 60.72
ld71r18u44fws graphics processing unit average utilisation percent = 54.12
ld71r16u13ws number of occurrences central processing unit went over 80percent  = 1600
ld71r18u44ews graphics processing unit average utilisation percent = 59.99
ld71r18u44cws core maximum utilisation percent (single core of highest usage) = 76.67
ld71r16u13ws graphics processing unit average utilisation percent = 39.03
ld71r18u44bws core maximum utilisation percent (single core of highest usage) = 70.21
ld71r16u15ws graphics processing unit average utilisation percent = 70.57
ld71r16u15ws core maximum utilisation percent (single core of highest usage) = 63.04
ld71r16u14ws graphics processing unit average utilisation percent = 87.0
ld71r16u13ws core maximum utilisation percent (single core of highest usage) = 99.87
ld71r18u44fws core maximum utilisation percent (single core of highest usage) = 96.48
ld71r18u44cws central processing unit total seconds over 80percent  = 120
ld71r16u14ws core maximum utilisation percent (single core of highest usage) = 53.59
ld71r18u44dws core maximum utilisation percent (single core of highest usage) = 67.71
ld71r16u14ws number of occurrences central processing unit went over 80percent  = 0
ld71r18u44fws central processing unit total seconds over 80percent  = 600
ld71r18u44cws number of occurrences central processing unit went over 80percent  = 1
ld71r18u44dws central processing unit total seconds over 80percent  = 0
    """
    return question_3_provided_knowledge

question_3_provided_knowledge = question_3_knowledge_provided()

def question_4_knowledge_provided():
    question_4_provided_knowledge = f"""
    Here is your context for a question I will ask you:
    If the central processing unit maximum utilisation percent is above 95%, it indicates that the machine is close to performing at maximum compute power which means it may be causing a bottleneck. However, if the average is low, then by itself it is not a reason to change the compute power.  
If the graphics processing unit maximum utilisation percent is above 95%, it indicates that the machine is close to performing at maximum compute power which means it may be causing a bottleneck. However, if the average is low, then by itself it is not a reason to change the compute power. 
If the central processing unit average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like being moved to a higher compute power with more cores.  Around 75% utlilisation is very efficient as it is making the most of the resources (good for sustainability), yet work is not slowed down by compute limitations.
If the graphics processing unit average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like a higher compute power with more cores. Around 75% utlilisation is very efficient as it is making the most of the resources (good for sustainability), yet work is not slowed down by compute limitations.
If the central processing unit average utilisation percent is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.
If the graphics processing unit average utilisation percent is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.
Sustainable Impact is a business imperative to HP. Reducing waste and carbon footprint is very important. Making sure machines run efficiently is important to reducing carbon emissions.
ld71r18u44ews central processing unit maximum utilisation percent = 79.78
ld71r18u44fws central processing unit maximum utilisation percent = 97.69
ld71r18u44bws central processing unit maximum utilisation percent = 64.72
ld71r18u44cws central processing unit maximum utilisation percent = 86.76
ld71r18u44dws central processing unit maximum utilisation percent = 71.96
ld71r16u14ws central processing unit maximum utilisation percent = 78.9
ld71r16u15ws central processing unit maximum utilisation percent = 66.69
ld71r18u44ews central processing unit average utilisation percent = 69.82
ld71r16u13ws central processing unit maximum utilisation percent = 97.51
ld71r18u44cws central processing unit average utilisation percent = 57.09
ld71r18u44bws central processing unit average utilisation percent = 29.45
ld71r18u44fws central processing unit average utilisation percent = 84.37
ld71r16u15ws central processing unit average utilisation percent = 20.1
ld71r16u13ws central processing unit average utilisation percent = 89.13
ld71r18u44dws central processing unit average utilisation percent = 30.08
ld71r16u14ws central processing unit average utilisation percent = 38.14
ld71r18u44ews graphics processing unit average utilisation percent = 59.99
ld71r18u44bws number of cores = 24
ld71r18u44cws number of cores = 24
ld71r18u44cws graphics processing unit average utilisation percent = 48.01
ld71r16u15ws number of cores = 28
ld71r16u14ws number of cores = 28
ld71r18u44ews number of cores = 24
ld71r18u44fws number of cores = 24
ld71r18u44ews core average utilisation percent (single core of highest usage) = 61.39
ld71r18u44bws graphics processing unit maximum utilisation percent = 95.03
ld71r18u44bws graphics processing unit average utilisation percent = 60.72
ld71r18u44dws graphics processing unit average utilisation percent = 60.99
ld71r16u13ws number of cores = 28
ld71r18u44dws number of cores = 24
ld71r18u44cws core maximum utilisation percent (single core of highest usage) = 76.67
ld71r18u44fws graphics processing unit average utilisation percent = 54.12
ld71r18u44cws core average utilisation percent (single core of highest usage) = 55.98
ld71r18u44ews core maximum utilisation percent (single core of highest usage) = 75.66
ld71r16u15ws core maximum utilisation percent (single core of highest usage) = 63.04
ld71r18u44dws core maximum utilisation percent (single core of highest usage) = 67.71
ld71r18u44bws core maximum utilisation percent (single core of highest usage) = 70.21
ld71r16u14ws core maximum utilisation percent (single core of highest usage) = 53.59
ld71r18u44fws core average utilisation percent (single core of highest usage) = 83.92
ld71r18u44bws core average utilisation percent (single core of highest usage) = 34.79
ld71r18u44fws core maximum utilisation percent (single core of highest usage) = 96.48
ld71r16u15ws graphics processing unit average utilisation percent = 70.57
ld71r16u15ws graphics processing unit maximum utilisation percent = 69.35
ld71r18u44dws core average utilisation percent (single core of highest usage) = 20.49
ld71r16u13ws core maximum utilisation percent (single core of highest usage) = 99.87
ld71r16u14ws core average utilisation percent (single core of highest usage) = 24.68
ld71r18u44fws graphics processing unit maximum utilisation percent = 46.9
ld71r16u15ws core average utilisation percent (single core of highest usage) = 18.1"""
    return question_4_provided_knowledge

question_4_provided_knowledge = question_4_knowledge_provided()

def question_5_knowledge_provided():
    question_5_provided_knowledge = f"""
    Here is your context for a question I will ask you:
    ld71r18u44bws central processing unit average utilisation percent = 29.45
ld71r18u44fws central processing unit average utilisation percent = 84.37
ld71r18u44cws central processing unit average utilisation percent = 57.09
ld71r18u44dws central processing unit average utilisation percent = 30.08
ld71r18u44ews central processing unit average utilisation percent = 69.82
ld71r16u15ws central processing unit average utilisation percent = 20.1
If the central processing unit average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like being moved to a higher compute power with more cores.  Around 75% utlilisation is very efficient as it is making the most of the resources (good for sustainability), yet work is not slowed down by compute limitations.
ld71r16u13ws central processing unit average utilisation percent = 89.13
ld71r16u14ws central processing unit average utilisation percent = 38.14
If the central processing unit average utilisation percent is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.
If the central processing unit maximum utilisation percent is above 95%, it indicates that the machine is close to performing at maximum compute power which means it may be causing a bottleneck. However, if the average is low, then by itself it is not a reason to change the compute power.  
ld71r18u44fws central processing unit maximum utilisation percent = 97.69
ld71r18u44dws central processing unit maximum utilisation percent = 71.96
ld71r18u44bws central processing unit maximum utilisation percent = 64.72
ld71r18u44cws central processing unit maximum utilisation percent = 86.76
ld71r16u14ws central processing unit maximum utilisation percent = 78.9
ld71r16u15ws central processing unit maximum utilisation percent = 66.69
ld71r16u13ws central processing unit maximum utilisation percent = 97.51
ld71r18u44ews central processing unit maximum utilisation percent = 79.78
If the graphics processing unit average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like a higher compute power with more cores. Around 75% utlilisation is very efficient as it is making the most of the resources (good for sustainability), yet work is not slowed down by compute limitations.
ld71r18u44cws core average utilisation percent (single core of highest usage) = 55.98
ld71r18u44bws core average utilisation percent (single core of highest usage) = 34.79
ld71r18u44cws graphics processing unit average utilisation percent = 48.01
If the graphics processing unit average utilisation percent is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.
ld71r18u44fws graphics processing unit average utilisation percent = 54.12
ld71r18u44bws graphics processing unit average utilisation percent = 60.72
ld71r18u44fws core average utilisation percent (single core of highest usage) = 83.92
ld71r18u44dws graphics processing unit average utilisation percent = 60.99
ld71r18u44dws core average utilisation percent (single core of highest usage) = 20.49
ld71r16u15ws core average utilisation percent (single core of highest usage) = 18.1
ld71r16u14ws core average utilisation percent (single core of highest usage) = 24.68
ld71r16u13ws core average utilisation percent (single core of highest usage) = 90.46
ld71r18u44ews core average utilisation percent (single core of highest usage) = 61.39
ld71r16u13ws graphics processing unit average utilisation percent = 39.03
ld71r16u15ws graphics processing unit average utilisation percent = 70.57
ld71r18u44ews graphics processing unit average utilisation percent = 59.99
ld71r16u14ws graphics processing unit average utilisation percent = 87.0
ld71r18u44bws central processing unit total seconds over 80percent  = 0
ld71r16u13ws central processing unit total seconds over 80percent  = 96000
ld71r16u14ws central processing unit total seconds over 80percent  = 0
ld71r16u15ws central processing unit total seconds over 80percent  = 0
ld71r18u44cws graphics processing unit average memory utilisation percent = 28.98
ld71r18u44fws graphics processing unit average memory utilisation percent = 13.45
ld71r18u44bws graphics processing unit average memory utilisation percent = 27.53
If the graphics processing unit maximum utilisation percent is above 95%, it indicates that the machine is close to performing at maximum compute power which means it may be causing a bottleneck. However, if the average is low, then by itself it is not a reason to change the compute power. 
ld71r18u44bws number of occurrences central processing unit went over 80percent  = 0
ld71r18u44dws graphics processing unit average memory utilisation percent = 29.44
ld71r16u14ws graphics processing unit average memory utilisation percent = 18.19
ld71r16u15ws core maximum utilisation percent (single core of highest usage) = 63.04
ld71r18u44cws core maximum utilisation percent (single core of highest usage) = 76.67
ld71r16u13ws core maximum utilisation percent (single core of highest usage) = 99.87
ld71r16u13ws graphics processing unit average memory utilisation percent = 31.61
ld71r18u44bws core maximum utilisation percent (single core of highest usage) = 70.21
ld71r16u15ws graphics processing unit average memory utilisation percent = 28.44
ld71r16u14ws core maximum utilisation percent (single core of highest usage) = 53.59
    
    """
    return question_5_provided_knowledge

question_5_provided_knowledge = question_5_knowledge_provided()

def question_6_knowledge_provided():
    question_6_provided_knowledge = f"""
    Here is your context for a question I will ask you:
    ld71r18u44dws carbon emissions (gco2eq) - use this for questions about carbon emissions = 613.84
ld71r16u13ws carbon emissions (gco2eq) - use this for questions about carbon emissions = 645.33
ld71r18u44bws carbon emissions (gco2eq) - use this for questions about carbon emissions = 614.19
ld71r18u44fws carbon emissions (gco2eq) - use this for questions about carbon emissions = 605.71
ld71r16u15ws carbon emissions (gco2eq) - use this for questions about carbon emissions = 774.84
ld71r18u44cws carbon emissions (gco2eq) - use this for questions about carbon emissions = 598.92
ld71r16u14ws carbon emissions (gco2eq) - use this for questions about carbon emissions = 842.96
ld71r18u44ews carbon emissions (gco2eq) - use this for questions about carbon emissions = 613.12
ld71r18u44bws operational carbon (gco2eq) = 107.48
ld71r18u44fws embodied carbon (gco2eq) = 506.7
ld71r18u44cws embodied carbon (gco2eq) = 506.7
ld71r18u44bws embodied carbon (gco2eq) = 506.7
ld71r16u14ws embodied carbon (gco2eq) = 418.9
ld71r16u13ws embodied carbon (gco2eq) = 418.9
ld71r18u44dws embodied carbon (gco2eq) = 506.7
ld71r16u15ws embodied carbon (gco2eq) = 418.9
ld71r16u14ws operational carbon (gco2eq) = 424.06
ld71r18u44cws operational carbon (gco2eq) = 92.22
ld71r18u44ews embodied carbon (gco2eq) = 506.7
ld71r18u44fws operational carbon (gco2eq) = 99.01
Operational carbon was produced over a total of 60.0 hours. Each day was 12 hours.
The embodied carbon for each machine is produced during the manufacturing and have already been proportionately distributed over the observed duration, using the Green Software Foundation's Impact Framework to calculate this. Carbon Emissions already includes embodied and operational. We must NEVER add embodied AND operational AND carbon as this will give us double the actual value.
ld71r16u13ws operational carbon (gco2eq) = 226.43
ld71r18u44dws operational carbon (gco2eq) = 107.13
ld71r16u15ws operational carbon (gco2eq) = 355.94
ld71r18u44ews operational carbon (gco2eq) = 106.41
Sustainable Impact is a business imperative to HP. Reducing waste and carbon footprint is very important. Making sure machines run efficiently is important to reducing carbon emissions.
ld71r18u44ews central processing unit maximum utilisation percent = 79.78
ld71r18u44bws central processing unit maximum utilisation percent = 64.72
ld71r18u44fws central processing unit maximum utilisation percent = 97.69
ld71r18u44cws central processing unit maximum utilisation percent = 86.76
ld71r16u15ws central processing unit maximum utilisation percent = 66.69
ld71r16u14ws central processing unit maximum utilisation percent = 78.9
ld71r18u44dws central processing unit maximum utilisation percent = 71.96
ld71r16u13ws central processing unit maximum utilisation percent = 97.51
ld71r16u15ws core maximum utilisation percent (single core of highest usage) = 63.04
ld71r18u44bws core maximum utilisation percent (single core of highest usage) = 70.21
ld71r16u13ws core maximum utilisation percent (single core of highest usage) = 99.87
ld71r18u44cws core maximum utilisation percent (single core of highest usage) = 76.67
ld71r16u14ws core maximum utilisation percent (single core of highest usage) = 53.59
ld71r18u44dws core maximum utilisation percent (single core of highest usage) = 67.71
ld71r18u44ews core maximum utilisation percent (single core of highest usage) = 75.66
ld71r18u44fws core maximum utilisation percent (single core of highest usage) = 96.48
ld71r18u44fws central processing unit average utilisation percent = 84.37
ld71r18u44ews central processing unit average utilisation percent = 69.82
ld71r16u13ws central processing unit average utilisation percent = 89.13
ld71r18u44bws central processing unit average utilisation percent = 29.45
ld71r18u44cws central processing unit average utilisation percent = 57.09
If the central processing unit maximum utilisation percent is above 95%, it indicates that the machine is close to performing at maximum compute power which means it may be causing a bottleneck. However, if the average is low, then by itself it is not a reason to change the compute power.  
ld71r16u15ws central processing unit average utilisation percent = 20.1
ld71r18u44dws central processing unit average utilisation percent = 30.08
ld71r18u44cws core average utilisation percent (single core of highest usage) = 55.98
ld71r18u44ews core average utilisation percent (single core of highest usage) = 61.39
ld71r16u14ws central processing unit average utilisation percent = 38.14
ld71r18u44fws core average utilisation percent (single core of highest usage) = 83.92
    """
    return question_6_provided_knowledge

question_6_provided_knowledge = question_6_knowledge_provided()

# Create base data
base_data_set_1 = {
    'question': [
        'Can you tell me how much carbon emission is produced by machine ld71r18u44dws?',
        'How much is the total carbon emissions for all the 8 machines?',
        'Which machine has the GPU highest average utilisation?',
        'Give me a summary of the central processing unit usage for all the machines?',
        'Which of the machines do you recommend being moved to the next level up of compute power and why?',
        'What is the central processing unit average utilisation for each machine?',
        'What machine has the highest carbon emission value?'
    ],
    'ideal_response': [
        f'The carbon emissions for machine 1d71r18u44dws are {u44dws_emissions} gCO2eq.',
        f'The total carbon emissions for the machines is {total_emissions} gCO2eq. This value includes both operational and embodied carbon emissions for all machines.',
        f'The machine with the highest GPU average is {machine_with_max_gpu} with {max_gpu_util}%.',
        f'''{cpu_utilization_str}

        The average CPU utilisation across the machines is about {average_cpu_util}%. The machine with the highest average CPU utilisation is {machine_with_max_cpu} with {max_cpu_util}%. The machine with the lowest average CPU utilisation is {machine_with_min_cpu} with {min_cpu_util}%.

        The CPU highest maximum utilisation is {machine_with_max_cpu_max} with {max_cpu_util_max}%. The CPU lowest maximum utilisation is {machine_with_min_cpu_max} with {min_cpu_util_max}%.
        The machine using its CPU most intensively is {machine_with_max_cpu_seconds_over_80} with {max_cpu_seconds_over_80} seconds over 80%.

        ''',
        f'''Machine {machine_with_max_cpu_seconds_over_80} should be moved to the next level up of compute power because it has experienced prolonged periods of high CPU utilisation, with a total of {max_cpu_seconds_over_80} seconds over 80%. 
        With an average CPU utilisation of {machine_with_max_cpu_seconds_over_80_avg_cpu}%, it is under heavy load and may be limimting work.''',
        f'{cpu_utilization_str}',
        f'Machine {machine_with_max_emissions} has the highest carbon emission value of {max_emissions} gCO2eq. This is made up of both operational and embodied carbon emissions.',
       
        
    ],
    'provided_knowledge': [
        [f'{question_0_provided_knowledge}'],
       [f'{question_1_provided_knowledge}'],
        [f'{question_2_provided_knowledge}'],
        [f'{question_3_provided_knowledge}'],
        [f'{question_4_provided_knowledge}'],
        [f'{question_5_provided_knowledge}'],
        [f'{question_6_provided_knowledge}']
    ]
}


# Create a list to hold all rows
rows = []
llm_responses_set_1 = [[
    
    'According to the provided data, the carbon emissions (gco2eq) produced by machine "ld71r18u44dws" are 613.84.',

    'According to the provided data, the carbon emissions (gco2eq) produced by machine "ld71r18u44dws" are 613.84.',

    'According to the provided data, the carbon emissions (gco2eq) produced by machine "ld71r18u44dws" are 613.84.',

    '''According to the provided data, the carbon emissions (gco2eq) produced by machine "ld71r18u44dws" are:
613.84
There is no additional relevant data that I think is important to add to this answer. The above answer provides the necessary information about the carbon emissions produced by the machine.''',

'According to the provided data, the carbon emissions produced by machine "ld71r18u44dws" are 613.84 gco2eq.',

'''According to the provided data, the carbon emissions (gco2eq) for machine "ld71r18u44dws" are:
613.84
So, the answer is: 613.84 gCO2eq.
There is no additional relevant data that I think is important to add to this answer.''',

'According to the provided data, the carbon emissions (gco2eq) produced by machine "ld71r18u44dws" are 613.84.',

'''According to the provided data, the carbon emissions (gco2eq) for machine "ld71r18u44dws" are:
613.84
So, the answer is: 613.84 gCO2eq.''',

''' According to the provided data, the carbon emissions (gCO2eq) produced by machine "ld71r18u44dws" are 613.84.
So, the answer is: 613.84 gCO2eq.''',

''' According to the provided data, the carbon emissions (gco2eq) produced by machine "ld71r18u44dws" are:
613.84 gCO2eq.
There is no additional relevant data that I think is important to add to answer this question.'''],

                       ["""Answer in gcO2eq: 5308.91
The total carbon emissions for all the 8 machines is already provided: 5308.91 gCO2eq. This value includes both operational and embodied carbon emissions.
It\'s worth noting that the data provides additional information about the carbon emissions, such as the operational carbon emissions produced during a specific period (60 hours) and the embodied carbon emissions associated with the manufacturing process of each machine. 
However, since you are asking for the total carbon emissions for all 8 machines, the provided value of 5308.91 gCO2eq is sufficient to answer your question.""",

'''Answer in gcO2eq: 5308.91
 The total carbon emissions for all 8 machines is already provided and that is 5308.91 gCO2eq. This includes both operational and embodied carbon emissions.''',

'''Answer in gcO2eq: 5308.91
The total carbon emissions for all the 8 machines is already provided as 5308.91 gCO2eq. This value includes both operational and embodied carbon emissions.''',

'''Answer in gcO2eq: 5308.91

The total carbon emissions for all the 8 machines is already provided as 5308.91 gCO2eq. This answer takes into account both operational and embodied carbon emissions, which are already combined. 
There is no additional relevant data to enhance this answer. The provided information seems to focus on individual machine performance metrics such as central processing unit average and maximum utilisation percentages, 
core average utilisation percentages, and graphics processing unit average utilisation percentages. These metrics do not affect the total carbon emissions calculation for all 8 machines, which is already given as 5308.91 gCO2eq.''',

'''Answer in gcO2eq: 5308.91
 The total carbon emissions for all the 8 machines is already provided as 5308.91 gCO2eq. This answer takes into account both operational and embodied carbon emissions, which are already combined to provide a comprehensive picture of the machines\' environmental impact.''', 

 '''Answer in gcO2eq: 5308.91
The total carbon emissions for all the 8 machines is 5308.91 gCO2eq. This value already includes both embodied and operational carbon emissions, which were calculated using the Green Software Foundation\'s Impact Framework.''' ,

'''Answer in gcO2eq: 5308.91
The total carbon emissions for all the 8 machines is 5308.91 gCO2eq. This value already includes both embodied and operational carbon emissions, which were calculated using the Green Software Foundation\'s Impact Framework.''',

'''Answer in gcO2eq: 5308.91

The total carbon emissions for all the 8 machines is 5308.91 gCO2eq. This value already includes both embodied and operational carbon emissions, which were calculated using the Green Software Foundation\'s Impact Framework.''',
'''Answer in gcO2eq: 5308.91

The total carbon emissions for all the 8 machines is 5308.91 gCO2eq. This value already includes both embodied and operational carbon emissions, which were calculated using the Green Software Foundation\'s Impact Framework.''',

'''Answer in gcO2eq: 5308.91
The total carbon emissions for all the 8 machines is 5308.91 gCO2eq. This value already includes both embodied and operational carbon emissions, which were calculated using the Green Software Foundation\'s Impact Framework.'''],

                       ['''To find the machine with the GPU highest average utilization, I\'ll compare the "graphics processing unit average utilisation percent" values across all 8 machines.
Here are the values:
* ld71r18u44bws: 60.72
* ld71r18u44cws: 48.01
* ld71r18u44fws: 54.12
* ld71r18u44ews: 59.99
* ld71r18u44dws: 60.99
* ld71r16u13ws: 39.03
* ld71r16u14ws: 87.0
* ld71r16u15ws: 70.57
The machine with the highest average GPU utilization is **ld71r18u44dws** with a value of **60.99%**.
Note that there are no additional relevant data points beyond what\'s provided, so my answer will remain the same as before.''',

'''To find the machine with the GPU highest average utilization, I\'ll compare the "graphics processing unit average utilisation percent" values across all 8 machines.
Here are the values:
* ld71r16u15ws: 70.57%
* ld71r18u44fws: 54.12%
* ld71r18u44dws: 60.99%
* ld71r16u14ws: 87.0%

The machine with the highest average GPU utilization is... **ld71r16u14ws** with a value of **87.0%**!''',


''' To find the machine with the GPU highest average utilization, I\'ll compare the "graphics processing unit average utilisation percent" values across all 8 machines.
Here are the values:

* ld71r18u44bws: 60.72
* ld71r18u44cws: 48.01
* ld71r18u44fws: 54.12
* ld71r18u44dws: 60.99
* ld71r16u13ws: 39.03
* ld71r16u14ws: 87.0
* ld71r16u15ws: 70.57
* ld71r18u44ews: 59.99

The machine with the highest average GPU utilization is **ld71r16u14ws** with a value of **87.0%**.
Note that there are no additional relevant data to enhance this answer, so I will not add any further information.''',


'''To find the machine with the GPU highest average utilization, I\'ll compare the "graphics processing unit average utilisation percent" values across all 8 machines.
The machine with the highest average GPU utilization is **ld71r16u14ws** with a value of **87.0%**.
There is no additional relevant data that can enhance this answer.''', 

'''To find the machine with the GPU highest average utilization, I\'ll compare the "graphics processing unit average utilisation percent" values across all 8 machines.
Here are the values:

* ld71r16u14ws: 87.0
* ld71r18u44ews: 59.99
* ld71r18u44bws: 60.72
* ld71r18u44cws: 48.01
* ld71r18u44dws: 60.99
* ld71r16u15ws: 70.57
* ld71r16u13ws: 39.03
* ld71r18u44fws: 54.12

The machine with the GPU highest average utilization is **ld71r16u14ws**, with a value of **87.0%**.
There is no additional relevant data that I think is important to add to my answer, so I will not provide any further information.''', 


'''To find the machine with the GPU highest average utilization, I\'ll compare the "graphics processing unit average utilisation percent" values across all 8 machines.
The machine with the highest average GPU utilization is **ld71r16u14ws** with a value of **87.0%**.
There is no additional relevant data that suggests any other machine has a higher average GPU utilization.''',


'''To find the machine with the GPU highest average utilization, I\'ll compare the "graphics processing unit average utilisation percent" values across all 8 machines
The values are:
* ld71r16u15ws: 70.57%
* ld71r18u44bws: 60.72%
* ld71r18u44cws: 48.01%
* ld71r18u44fws: 54.12%
* ld71r18u44ews: 59.99%
* ld71r16u13ws: 39.03% (not the highest)
* ld71r16u14ws: 87.0% (the highest!)
* ld71r18u44dws: 60.99%
The machine with the GPU highest average utilization is **ld71r16u14ws**, with a value of **87.0%**.
There is no additional relevant data that I think is important to add to answer this question, so my answer will remain as above.''',


'''To find the machine with the GPU highest average utilization, I\'ll compare the "graphics processing unit average utilisation percent" values across all 8 machines.
Here are the values:

* ld71r18u44cws: 48.01
* ld71r18u44ews: 59.99
* ld71r16u13ws: 39.03
* ld71r16u15ws: 70.57
* ld71r16u14ws: 87.0
* ld71r18u44dws: 60.99
* ld71r18u44fws: 54.12
* ld71r18u44bws: 60.72

The machine with the highest average GPU utilization is... **ld71r16u14ws** with a value of **87.0%!**

So, according to this data, the machine "ld71r16u14ws" has the GPU with the highest average utilisation.

Note that there are no additional relevant data points beyond what\'s already been presented in terms of determining which machine has the GPU with the highest average utilization.''',

'''To find the machine with the GPU highest average utilization, I\'ll compare the "graphics processing unit average utilisation percent" values across all 8 machines.

Here are the values:

* ld71r18u44bws: 60.72%
* ld71r18u44cws: 48.01%
* ld71r18u44fws: 54.12%
* ld71r18u44ews: 59.99%
* ld71r18u44dws: 60.99%
* ld71r16u13ws: 39.03%
* ld71r16u15ws: 70.57%

The machine with the highest average GPU utilization is... **ld71r16u15ws** with a value of **70.57%**!''',

'''To find the machine with the GPU highest average utilization, I\'ll compare the "graphics processing unit average utilisation percent" values across all 8 machines.

The machine with the GPU highest average utilization is **ld71r16u15ws** with a value of **70.57%**.

Note that ld71r18u44bws has a high GPU maximum utilisation percent at 95.03%, which indicates that the machine may be close to performing at maximum compute power, potentially causing a bottleneck. However, this does not affect our answer as we are looking for the average utilization. '''],
                     
                       ['''Based on the provided data, here\'s a summary of the central processing unit (CPU) usage for all the machines:

The average CPU utilization across all machines is approximately 52.5%. This suggests that most machines are operating at a moderate level.

The maximum CPU utilization observed was around 97.691% in machine "ld71r18u44fws". This indicates that this machine is under heavy load and may need more resources, such as being moved to a higher compute power with more cores.

Four machines ("ld71r16u13ws", "ld71r18u44fws", "ld71r18u44cws", and "ld71r18u44ews") have spent a significant amount of time (total seconds over 80%) with high CPU utilization. The total seconds over 80% range from 0 to 96000.

The core utilization average is approximately 43.5%. This suggests that most machines are operating at a moderate level on their individual cores.

One machine ("ld71r16u13ws") has experienced prolonged periods of high CPU utilization, with a total of 96000 seconds over 80%.''',


'''Based on the provided data, here\'s a summary of the central processing unit (CPU) usage for all machines:

The average CPU utilization across all machines is 54.45%. The maximum CPU utilization observed was 97.69% in machine ld71r18u44fws.

Machines that spent significant time above 80% CPU utilization are:

* ld71r16u13ws (96000 seconds)
* ld71r18u44fws (600 seconds)
* ld71r18u44cws (120 seconds)

ld71r18u44bws, on the other hand, had a low CPU utilization with an average of 29.45% and no instances above 80%.

ld71r18u44ews had an average CPU utilization of 69.82%, while ld71r16u15ws and ld71r16u14ws had lower utilizations at 20.1% and 38.14%, respectively.

It\'s worth noting that if the central processing unit average utilization percent is above 85%, it may indicate that the machine is under heavy load and may need more resources, like being moved to a higher compute power with more cores. Conversely, if the CPU utilization is below 15%, it may be a sign of underutilization and potential cost savings by reducing resources.

The maximum CPU utilization for each machine varies, but ld71r16u13ws and ld71r18u44fws both reached above 95%.''',


'''Based on the provided data, here\'s a summary of the central processing unit (CPU) usage for all 8 machines:

The average CPU utilization across all 8 machines is 54.45%. The maximum CPU utilization was observed in machine ld71r18u44fws at 97.69%.

Machines with high CPU utilization include ld71r18u44fws, which had an average CPU utilization of 84.37%, and ld71r16u13ws, which had an average CPU utilization of 89.13%. Machines ld71r18u44cws and ld71r16u14ws had moderate CPU utilization, ranging from 55% to 78%.

Machine ld71r18u44fws spent a total of 600 seconds (approximately 16 minutes) with CPU utilization above 80%, while machine ld71r16u13ws spent a total of 96000 seconds (approximately 27 hours) with CPU utilization above 80%.

Overall, the data suggests that machines ld71r18u44fws and ld71r16u13ws have higher CPU utilization compared to the others.

No additional relevant data is provided.''',

'''Based on the provided data, here\'s a summary of the central processing unit (CPU) usage for all the machines:

* Average CPU utilization: The average CPU utilization across all machines is approximately 54.45% (calculated by adding up the average CPU utilization percentages and dividing by the number of machines).
* Maximum CPU utilization: The maximum CPU utilization across all machines is approximately 97.69% (observed in machine "ld71r18u44fws").
* Time spent over 80% utilization: A total of 106,600 seconds (approximately 30 hours) were spent by the CPUs on all machines with utilization above 80%.
* Machines with high CPU usage:
        + Machine "ld71r16u13ws" had a CPU utilization above 80% for approximately 96,000 seconds.
        + Machine "ld71r18u44fws" had a CPU utilization above 80% for approximately 600 seconds.

Newly added data:

* ld71r18u44bws central processing unit average utilisation percent = 29.45
* ld71r16u13ws central processing unit total seconds over 80percent  = 96000

The newly added data shows that machine "ld71r18u44bws" has a relatively low CPU utilization of approximately 29.45%, indicating that it is underutilized and could potentially benefit from reducing resources or moving to a lower compute power.

On the other hand, machine "ld71r16u13ws" spent a significant amount of time (approximately 96,000 seconds) with CPU utilization above 80%, indicating that it may be experiencing heavy load and potentially needing more resources or being moved to a higher compute power.''',


'''Based on the provided data, here\'s a summary of the central processing unit (CPU) usage for all the machines:

The average CPU utilization across all machines is approximately 54.45%. The maximum CPU utilization observed was 97.69% in machine ld71r18u44fws.

Machine ld71r16u13ws spent significant time above 80% CPU utilization, with a total of 96000 seconds (approximately 26.67 hours) being spent at this level.

The average core utilization across all machines is approximately 44.45%. The maximum core utilization observed was 99.87% in machine ld71r16u13ws.

Overall, the CPU usage varies significantly across different machines, with some machines experiencing high utilization rates while others remain relatively idle.''',


'''Based on the provided data, here\'s a summary of the central processing unit (CPU) usage for all the machines:

The average CPU utilization across all machines is approximately 54.45%. The maximum CPU utilization across all machines is around 97.69% observed in machine "ld71r16u13ws".

Machines that spent significant time above 80% CPU utilization include:
- "ld71r16u13ws" with a total of 96000 seconds (approximately 27.4 hours).
- "ld71r18u44fws" with a total of 600 seconds.

Notable trends observed are:

* Machines like "ld71r18u44cws", "ld71r18u44ews", and "ld71r16u13ws" have relatively high CPU utilization, indicating potential resource bottlenecks.
* On the other hand, machines like "ld71r18u44bws", "ld71r18u44dws", and "ld71r16u15ws" have lower CPU utilization, suggesting they may be underutilized or experiencing less load.

It is important to note that if the central processing unit average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like being moved to a higher compute power with more cores.''',


'''Based on the provided data, here\'s a summary of the central processing unit (CPU) usage for all the machines:

The average CPU utilization across all machines is approximately 54.45%. The maximum CPU utilization observed in any machine is approximately 97.69% in machine ld71r16u13ws.

Machine ld71r18u44fws has a high CPU utilization of around 84.37%, indicating that it may be under heavy load and require more resources to optimize performance. On the other hand, machines like ld71r18u44bws and ld71r18u44dws have low CPU utilizations below 15%, suggesting that they are underutilized and could benefit from reduced resources.

The total seconds spent with CPU utilization above 80% across all machines is approximately 96000 seconds (observed in machine ld71r16u13ws), which accounts for about 26.4% of the total possible time.

Overall, the CPU usage varies significantly across different machines, with some machines experiencing high utilization rates while others remain relatively idle.''',

'''Based on the provided data, here\'s a summary of the central processing unit (CPU) usage for all the machines:

* Average CPU utilization: The average CPU utilization across all machines is approximately 54.5% (calculated by adding up the average CPU utilization percentages and dividing by the number of machines).
* Maximum CPU utilization: The maximum CPU utilization across all machines is approximately 97.69% (observed in machine "ld71r18u44fws").
* Time spent over 80% utilization: A total of 106,600 seconds (approximately 30 hours) were spent by the CPUs on all machines with utilization above 80%.
* Machines with high CPU usage:
        + Machine "ld71r16u13ws" had a maximum CPU utilization of 97.51% and spent 96,000 seconds (approximately 26.67 hours) with utilization above 80%.
        + Machine "ld71r18u44fws" had a maximum CPU utilization of 97.69% and spent 600 seconds (approximately 0.17 hours) with utilization above 80%.

Additional insights:

* Machine "ld71r18u44bws" has relatively low CPU usage, with an average utilization of 29.45% and no instances of CPU utilization exceeding 80%.
* Machines "ld71r16u13ws", "ld71r16u14ws", and "ld71r18u44fws" have high CPU usage, with average utilizations ranging from 89.13% to 84.37%. These machines spent a significant amount of time with utilization above 80%.
* Machines "ld71r18u44cws" and "ld71r16u15ws" have moderate CPU usage, with average utilizations ranging from 57.09% to 20.1%.

Overall, the data suggests that machines "ld71r16u13ws", "ld71r16u14ws", and "ld71r18u44fws" are experiencing high CPU utilization, while machine "ld71r18u44bws" is underutilized.''',

'''Based on the provided data, here\'s a summary of the central processing unit (CPU) usage for all the machines:

The average CPU utilization across all machines is approximately 54.45%. The maximum CPU utilization across all machines is approximately 97.69% observed in machine ld71r18u44fws.

Machine-wise breakdown shows that:

* Machine ld71r18u44ews has an average CPU utilization of 69.82% and a maximum CPU utilization of 79.78%.
* Machine ld71r18u44dws has an average CPU utilization of 30.08% and a maximum CPU utilization of 71.96%.
* Machine ld71r18u44bws has an average CPU utilization of 29.45% and a maximum CPU utilization of 64.72%.
* Machine ld71r18u44cws has an average CPU utilization of 57.09% and a maximum CPU utilization of 86.76%.
* Machine ld71r16u15ws has an average CPU utilization of 20.1% and a maximum CPU utilization of 66.69%.
* Machine ld71r16u14ws has an average CPU utilization of 38.14% and a maximum CPU utilization of 78.9%.
* Machine ld71r18u44fws has an average CPU utilization of 84.37% and a maximum CPU utilization of 97.69%.
* Machine ld71r16u13ws has an average CPU utilization of 89.13% and a maximum CPU utilization of 97.51%.

If the central processing unit (CPU) average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like being moved to a higher compute power with more cores.

If the central processing unit (CPU) average utilisation percent is below 15%, it indicates that the machine is underutilised and it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the 
business.

Additionally, machines ld71r18u44fws and ld71r16u13ws have spent significant time with CPU utilization above 80%, indicating heavy load on these machines.''',

'''Based on the provided data, here\'s a summary of the central processing unit (CPU) usage for all the machines:

The average CPU utilization across all machines is approximately 54.5%. The maximum CPU utilization across all machines is approximately 97.69% (observed in machine ld71r16u13ws and ld71r18u44fws).

Machines that spent significant time above 80% CPU utilization include:

* ld71r16u13ws (96000 seconds)
* ld71r18u44fws (600 seconds)
* ld71r18u44cws (120 seconds)

It\'s worth noting that ld71r18u44fws has a high average CPU utilization of 84.37%, which indicates it may be under heavy load and could benefit from more resources, such as being moved to a higher compute power with more cores.

On the other hand, ld71r16u15ws has a low average CPU utilization of 20.1%, indicating that it is underutilized and could potentially reduce its resources, like using a lower compute power, to save costs and be more environmentally friendly.

The remaining machines have average CPU utilizations ranging from 29.45% to 69.82%.'''],


                       ['''Based on the provided data, I would recommend moving machine "ld71r18u44fws" to the next level up of compute power. Here'\s why:

The central processing unit (CPU) average utilization for this machine is 84.37%, which indicates that it is under heavy load and may need more resources. Additionally, the CPU maximum utilization is 97.69%, which suggests that there are opportunities for further optimization or upgrading to take advantage of the available resources.

Furthermore, considering that the single core with the highest usage has a maximum utilization of 96.48%, this machine is close to performing at maximum compute power, which may be causing a bottleneck. Therefore, moving it to the next level up of compute power could help alleviate this issue and improve its overall performance.

In contrast, machines like "ld71r18u44bws" and "ld71r16u15ws" have lower CPU utilization rates (29.45% and 20.1%, respectively), which may indicate that they are not as heavily utilized and do not require an immediate upgrade.

Overall, considering the high CPU utilization and opportunities for optimization on machine "ld71r18u44fws", I recommend moving it to the next level up of compute power to take advantage of its available resources.''','''
 Based on the provided data, I would recommend moving machine "ld71r18u44fws" to the next level up of compute power. Here\'s why:

Firstly, its central processing unit (CPU) average utilization is 84.37%, which indicates that this machine is heavily utilized and may need more resources to handle its workload efficiently. Secondly, its CPU maximum utilization is 97.69%, which suggests that the current hardware is being 
pushed to its limits, making it a good candidate for an upgrade.

Furthermore, the GPU average utilization is 54.12%, which also indicates that this machine is heavily utilized in terms of graphical processing. This suggests that the machine may be experiencing performance bottlenecks and could benefit from increased compute power.

In contrast, other machines like "ld71r16u14ws" and "ld71r18u44dws" have lower CPU and core utilizations, suggesting that they may not require an immediate upgrade. Machines like "ld71r16u13ws" and "ld71r18u44cws" have high CPU utilization but relatively low GPU utilization, which might indicate that the current hardware is sufficient for their specific workloads.

Overall, considering all 8 machines, I believe that machine "ld71r18u44fws" would benefit the most from being moved to the next level up of compute power due to its high CPU and GPU utilizations.''',

'''Based on the provided data, I would recommend moving machine "ld71r18u44fws" to the next level up of compute power. Here\'s why:

The central processing unit maximum utilisation percent for this machine is 97.69%, which indicates that it is close to performing at maximum compute power and may be causing a bottleneck. Additionally, the average CPU utilisation is 84.37%, which suggests that the machine is under heavy load and could benefit from more powerful processing capabilities.

Furthermore, the core maximum utilisation percent for single core of highest usage is 96.48%, which also indicates that the current machine is already pushing its limits. Considering these high levels of CPU utilization and potential bottlenecks, I believe it would be beneficial to upgrade 
this machine to a higher-powered one.

In contrast, other machines like "ld71r16u15ws" have lower CPU utilisation (20.1%) and may not require an immediate upgrade. Similarly, machines with high graphics processing unit (GPU) utilisation, such as "ld71r18u44bws", might benefit from a GPU upgrade rather than a CPU upgrade.        

Overall, considering the high CPU utilisation, consistent performance, and potential for bottlenecks, I recommend moving machine "ld71r18u44fws" to the next level up of compute power.''',

''' Based on the provided data, I would recommend moving machine "ld71r16u13ws" to the next level up of compute power. Here\'s why:

1. High CPU utilization: The maximum CPU utilisation percentage for this machine is 97.51%, which indicates that it is heavily utilized and could benefit from an upgrade.
2. Consistent high performance: The average CPU utilisation percentage is also relatively high at 89.13%. This suggests that the machine is consistently performing well, making it a good candidate for an upgrade.
3. No apparent bottlenecks: Unlike some of the other machines, there are no obvious bottlenecks or areas where the machine could be improved without upgrading its compute power.

In contrast, while some of the other machines may have high CPU utilisation percentages, they also have lower average utilisation percentages and/or other factors that suggest they might not benefit as much from an upgrade. For example:

* Machine "ld71r18u44fws" has a high maximum CPU utilisation percentage (97.69%), but its average utilisation percentage is relatively low (84.37%). This suggests that the machine may be able to handle its current workload without needing an upgrade.
* Machine "ld71r16u14ws" has a moderate CPU utilisation percentage, but its core average and maximum utilisation percentages are relatively low. This suggests that it might not benefit as much from an upgrade.

Overall, based on the data provided, I believe that machine "ld71r16u13ws" is the best candidate for being moved to the next level up of compute power.''',

'''Based on the provided data, I would recommend moving machine "ld71r18u44fws" to the next level up of compute power. Here\'s why:

1. High CPU utilization: The average CPU utilization for this machine is 84.37%, which is significantly higher than the other machines. This suggests that it may be a good candidate for upgrading its compute power.
2. High core maximum utilisation percent (single core of highest usage): The core maximum utilisation percent for this machine is 96.48%, which is also higher than the other machines. This indicates that even individual cores are being heavily utilized, making it a good candidate for upgrade.
3. Low graphics processing unit average utilisation percent: With an average GPU utilization of only 54.12%, this machine may not be utilizing its current compute power to its full potential.

In contrast, while some other machines (e.g., "ld71r18u44cws" and "ld71r16u13ws") have high CPU maximum utilisation percent, their average CPU utilization is lower, suggesting that they might not need an upgrade as urgently. Additionally, machine "ld71r18u44fws" has a relatively low GPU utilization, which could be improved with a more powerful compute node.

Overall, considering the high CPU and core utilization, as well as the potential for improving GPU performance, I believe that moving machine "ld71r18u44fws" to the next level up of compute power would be a good decision.''',

''' Based on the provided data, I would recommend moving machine "ld71r18u44fws" to the next level up of compute power. Here\'s why:

The central processing unit (CPU) maximum utilisation percent for this machine is 97.69%, which indicates that it is close to performing at maximum compute power and may be causing a bottleneck. Additionally, the CPU average utilisation percent is high at 84.37%, suggesting that the machine is under heavy load and may need more resources.

Furthermore, the core maximum utilisation percent (single core of highest usage) for this machine is 96.48%, which also suggests that the current machine\'s cores are being pushed to their limits. This, combined with the low graphics processing unit (GPU) utilisation at 54.12%, makes it a good candidate for an upgrade.

In contrast, while some other machines may have high CPU utilisation or maximum core utilisation, they do not have the same level of CPU and core utilisation as "ld71r18u44fws". Therefore, I believe that this machine would be the best candidate for a CPU-focused upgrade.

Please note that this recommendation is based on the provided data and may not take into account other factors that could affect the machine\'s performance or workload.''',

''' Based on the provided data, I would recommend moving machine "ld71r18u44fws" to the next level up of compute power. Here\'s why:

The central processing unit average utilisation percent for this machine is 84.37%, which indicates that it is under heavy load and may need more resources, like being moved to a higher compute power with more cores. Additionally, the single core with the highest usage has a maximum utilisation of 96.48%, suggesting that the current processing capacity is close to its maximum.

In contrast, machines like "ld71r18u44bws" and "ld71r16u15ws" have lower CPU utilization (29.45% and 20.1%, respectively) and may not require an immediate upgrade. Machines with high GPU utilization, such as "ld71r18u44cws", might benefit from a GPU upgrade rather than a full compute power 
upgrade.

Overall, considering the entire dataset, I believe that machine "ld71r18u44fws" is the most likely candidate for a compute power upgrade due to its high CPU utilization and potential for increased processing demands.''',

'''Based on the provided data, I would recommend moving machine "ld71r16u13ws" to the next level up of compute power. Here's why:

1. High CPU utilization: The central processing unit maximum utilisation percent for this machine is 97.51%, which indicates that it is heavily utilized and could benefit from an upgrade.
2. Consistent high performance: The average CPU utilisation percent is also relatively high at 89.13%, suggesting that the machine is consistently performing well, making it a good candidate for an upgrade.
3. No apparent bottlenecks: Unlike some of the other machines, there are no obvious bottlenecks or areas where the machine could be improved without upgrading its compute power.

In contrast, while some of the other machines may have high CPU utilisation percentages, they also have lower average utilisation percentages, suggesting that they may not be as heavily utilized overall. Additionally, some machines have higher graphics processing unit (GPU) utilisation percentages, which might indicate that a GPU upgrade would be more beneficial than a CPU upgrade.

Overall, based on the data provided, I believe that machine "ld71r16u13ws" is the best candidate for being moved to the next level up of compute power.''',

'''  Based on the provided data, I would recommend moving the machine "ld71r16u13ws" to the next level up of compute power. Here's why:

The average CPU utilization for this machine is 89.13%, which is higher than the average CPU utilization for the other machines (84.37%). This suggests that the machine is under heavy load and may need more resources, like a higher compute power with more cores.

Additionally, the maximum CPU utilization for this machine is high at 97.51%, which indicates that the machine is close to performing at maximum compute power, which may be causing a bottleneck.

In contrast, while "ld71r18u44fws" has a high maximum CPU utilization, its average CPU utilization is relatively low at 84.37%. This suggests that it may not be as heavily utilized as some of the other machines, and therefore may not require an upgrade to the next level of compute power.

Overall, considering both average and maximum CPU utilizations, I believe that "ld71r16u13ws" is a good candidate for upgrading to the next level of compute power.''',

'''Based on the provided data, I would recommend moving machine "ld71r18u44fws" to the next level up of compute power. Here\'s why:

1. High CPU utilization: The average CPU utilization is 84.37%, which indicates that this machine is heavily utilized and could benefit from an upgrade.
2. High core maximum utilisation percent (single core of highest usage): The value of 96.48% suggests that the single core with the highest usage is already close to its maximum capacity, indicating a need for more processing power.
3. High graphics processing unit average utilisation percent: Although not as high as CPU utilization, the GPU average utilization of 54.12% indicates that this machine is also heavily utilized in terms of graphical processing.

In contrast, other machines like "ld71r18u44cws" and "ld71r16u14ws" have lower CPU utilization (57.09% and 38.14%, respectively) and might not require an immediate upgrade. Machines like "ld71r18u44dws" and "ld71r16u15ws" have relatively low CPU utilization, suggesting that they may not be 
as heavily utilized.

Machine "ld71r18u44ews" has a moderate CPU utilization (69.82%) but is still below the threshold for an immediate upgrade. Machine "ld71r18u44bws" has a low CPU utilization (29.45%), indicating that it might not require an upgrade at this time.

Overall, considering all 8 machines, I believe that machine "ld71r18u44fws" would benefit the most from being moved to the next level up of compute power due to its high CPU and GPU utilization.

As per the additional context provided, if the central processing unit average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources. In this case, machine "ld71r18u44fws" has an average CPU utilization of 84.37%, which suggests that it may indeed benefit from being moved to a higher compute power with more cores.

Additionally, considering the maximum utilisation percent for central processing unit (CPU) and graphics processing unit (GPU), we can see that machine "ld71r18u44fws" has CPU maximum utilization of 97.69% and GPU maximum utilization of 46.9%. This indicates that this machine is close to performing at maximum compute power, which may be causing a bottleneck.

Therefore, based on the provided data and additional context, I strongly recommend moving machine "ld71r18u44fws" to the next level up of compute power to ensure optimal performance and reduce potential bottlenecks.'''],
                       ['''Based on the provided data, here are some observations and recommendations for each machine:

1. ld71r18u44bws: The CPU average utilisation is 29.45%, which indicates that the machine is not heavily loaded. However, the maximum CPU utilisation is only 64.72%, which suggests that there may be some idle time available.
2. ld71r18u44fws: The CPU average utilisation is 84.37%, which indicates that the machine is under heavy load and may need more resources to handle the workload efficiently.
3. ld71r18u44cws: The CPU average utilisation is 57.09%, which suggests that the machine is moderately loaded. The maximum CPU utilisation is 86.76%, which indicates that there may be some bottleneck potential if the load increases further.
4. ld71r18u44dws: The CPU average utilisation is 30.08%, which indicates that the machine is not heavily loaded. However, the maximum CPU utilisation is 71.96%, which suggests that there may be some idle time available.
5. ld71r18u44ews: The CPU average utilisation is 69.82%, which indicates that the machine is moderately loaded. The maximum CPU utilisation is 79.78%, which suggests that there may be some bottleneck potential if the load increases further.
6. ld71r16u15ws: The CPU average utilisation is 20.1%, which indicates that the machine is underutilised and may benefit from reducing resources to save costs and reduce environmental impact.
7. ld71r16u13ws: The CPU average utilisation is 89.13%, which indicates that the machine is under heavy load and may need more resources to handle the workload efficiently.
8. ld71r16u14ws: The CPU average utilisation is 38.14%, which suggests that the machine is moderately loaded.

In general, it appears that machines with high CPU utilisations (above 85%) are likely to benefit from additional resources or load balancing, while machines with low CPU utilisations (below 15%) may be candidates for resource reduction or consolidation. Machines with moderate CPU utilisations (around 50-75%) appear to be well-balanced and may not require significant changes.

Note that these observations are based solely on the provided data and do not take into account any other factors that may be relevant to the machines\' performance or resource allocation.''',

''' Based on the provided data, here are the central processing unit (CPU) average utilisations for each machine:

1. ld71r18u44bws: 29.45%
2. ld71r18u44dws: 30.08%
3. ld71r16u14ws: Not available (null)
4. ld71r18u44fws: 84.37%
5. ld71r16u13ws: 89.13% (machine is under heavy load and may need more resources, like being moved to a higher compute power with more cores)
6. ld71r16u15ws: 20.1%
7. ld71r18u44ews: 69.82%
8. ld71r18u44cws: 57.09%

Note that for machines ld71r16u14ws and ld71r16u13ws, the CPU average utilisation is not available (null).''',

'''Based on the provided data, here are the central processing unit (CPU) average utilisations for each machine:

* ld71r18u44bws: 29.45%
* ld71r18u44dws: 30.08%
* ld71r16u15ws: 20.1%
* ld71r16u14ws: 38.14%
* ld71r18u44ews: 69.82%
* ld71r18u44cws: 57.09%
* ld71r18u44fws: 84.37%
* ld71r16u13ws: 89.13%

The data suggests that machines ld71r18u44fws, ld71r18u44ews, and ld71r16u13ws are under heavy load, with CPU utilisations above 85%. Machines ld71r18u44bws, ld71r18u44dws, and ld71r16u15ws have moderate utilisations between 20% to 69.82%, indicating a balance between resource utilization and efficiency.

Machines ld71r16u14ws and ld71r18u44cws have lower CPU utilisations below 40%, suggesting underutilization and potential cost savings by reducing resources.

Note that there is no additional relevant data beyond what is provided, so this answer will be empty.''',

'''Based on the provided data, here are some key observations and recommendations for each machine:

1. ld71r18u44bws: The CPU average utilisation is 29.45%, which indicates that this machine is underutilised. There is no reason to change the compute power at this time.
2. ld71r18u44dws: The CPU average utilisation is 30.08%, which also suggests that this machine is underutilised. No changes are needed.
3. ld71r16u15ws: The CPU average utilisation is 20.1%, indicating that this machine is significantly underutilised. It would be more cost-effective to reduce resources, such as lowering the compute power, to save money and reduce environmental impact.
4. ld71r18u44cws: The CPU average utilisation is 57.09%, which indicates that this machine is moderately loaded. No changes are needed at this time.
5. ld71r18u44ews: The CPU average utilisation is 69.82%, suggesting that this machine is heavily loaded. It may need more resources, such as a higher compute power with more cores, to handle the workload efficiently.
6. ld71r16u13ws: The CPU average utilisation is 89.13%, indicating that this machine is extremely heavily loaded. It needs more resources, like being moved to a higher compute power with more cores, to handle the workload effectively.

In general, if the CPU average utilisation is above 85%, it suggests that the machine is under heavy load and may need more resources. If below 15%, it indicates that the machine is underutilised and could benefit from reduced resources.''',

'''Based on the provided data, here are the central processing unit (CPU) average utilisations for each machine:

* ld71r16u14ws: 38.14%
* ld71r18u44fws: 84.37%
* ld71r16u13ws: Not available (null)
* ld71r18u44bws: 29.45%
* ld71r18u44ews: 69.82%
* ld71r18u44dws: 30.08%
* ld71r18u44cws: 57.09%
* ld71r16u15ws: 20.1%

If the CPU average utilisation is above 85%, it indicates that the machine is under heavy load and may need more resources, like being moved to a higher compute power with more cores.

Around 75% utlilisation is very efficient as it is making the most of the resources (good for sustainability), yet work is not slowed down by compute limitations.

If the CPU average utilisation is below 15%, it indicates that the machine is underutilised and it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.

Note: For machine ld71r16u13ws, the CPU average utilisation is not available (null).''',

'''Based on the provided CPU average utilisations, here are some observations:

* ld71r18u44fws (84.37%) and ld71r16u13ws (89.13%) indicate that these machines are under heavy load and may need more resources to handle the workload.
* ld71r16u14ws (38.14%) suggests that this machine is underutilised and could potentially reduce its resources to save costs and be more environmentally friendly.
* ld71r18u44ews (69.82%) and ld71r18u44cws (57.09%) are operating at moderate levels, indicating efficient use of resources without causing a bottleneck.

No further relevant data is provided beyond this point.''',

'''Based on the provided data, here are the central processing unit (CPU) average utilisations for each machine:

1. ld71r18u44bws: The CPU usage is 29.45%, indicating that this machine is underutilised and it would be more cost-effective to reduce resources.
2. ld71r18u44ews: The CPU usage is 69.82%, which indicates that the machine is efficient and not causing a bottleneck.
3. ld71r18u44dws: The CPU usage is 30.08%, indicating that this machine is also underutilised and could benefit from reducing resources.
4. ld71r16u15ws: The CPU usage is 20.1%, confirming that this machine is underutilised and it would be more cost-effective to reduce resources.
5. ld71r18u44cws: The CPU usage is 57.09%, which indicates that the machine is efficient but not causing a bottleneck.

Note that machines 6, 7, and 8 do not have complete information as there is no average utilisation data provided for these machines.

There are no additional relevant data to enhance this answer.''',

'''Based on the provided data, here are my observations and recommendations for each machine:

1. ld71r18u44bws: The CPU average utilisation is 29.45%, which indicates that this machine is underutilised. Considering its low CPU maximum utilisation (64.72%) and no occurrences of CPU usage above 80%, there is no need to increase resources for this machine.

2. ld71r18u44fws: The CPU average utilisation is 84.37%, which indicates that this machine is under heavy load. Its high CPU maximum utilisation (97.69%) also suggests that it may be causing a bottleneck. I recommend increasing resources or moving this machine to a higher compute power with more cores.

3. ld71r18u44cws: The CPU average utilisation is 57.09%, which indicates that this machine is moderately utilized. Its moderate CPU maximum utilisation (86.76%) suggests that it may not be causing a bottleneck, but increasing resources could still improve performance.

4. ld71r18u44dws: The CPU average utilisation is 30.08%, which indicates that this machine is underutilised. Considering its low CPU maximum utilisation (71.96%), there is no need to increase resources for this machine.

5. ld71r18u44ews: The CPU average utilisation is 69.82%, which indicates that this machine is moderately utilized. Its moderate CPU maximum utilisation (79.78%) suggests that it may not be causing a bottleneck, but increasing resources could still improve performance.

6. ld71r16u15ws: The CPU average utilisation is 20.1%, which indicates that this machine is underutilised. Considering its low CPU maximum utilisation (66.69%), there is no need to increase resources for this machine.

7. ld71r16u13ws: The CPU average utilisation is 89.13%, which indicates that this machine is under heavy load. Its high CPU maximum utilisation (97.51%) also suggests that it may be causing a bottleneck. I recommend increasing resources or moving this machine to a higher compute power with 
more cores.

8. ld71r16u14ws: The CPU average utilisation is 38.14%, which indicates that this machine is moderately utilized. Its moderate CPU maximum utilisation (78.9%) suggests that it may not be causing a bottleneck, but increasing resources could still improve performance.

No additional relevant data has been provided, so my answer will end here.''',

''' Based on the provided data, here are my findings:

**Machines with high CPU utilisations (> 85%)**

* ld71r16u13ws (89.13%): This machine is under heavy load and may need more resources, like being moved to a higher compute power with more cores.
* ld71r18u44fws (84.37%): This machine is also under heavy load and may benefit from additional resources.

**Machines with low CPU utilisations (< 15%)**

* ld71r16u15ws (20.1%): This machine is underutilised and it would be more cost-effective to reduce resources, like a lower compute power.
* ld71r16u14ws (38.14%): While not as low as 15%, this machine still has some room for improvement in terms of resource utilization.

**Machines with high CPU maximum utilisations (> 95%)**

* ld71r16u13ws (97.51%): This machine is close to performing at maximum compute power, which may cause a bottleneck.
* ld71r18u44fws (97.69%): Similarly, this machine is also approaching its maximum capacity.

**Other observations**

* ld71r18u44cws has a relatively efficient CPU utilisation of 57.09%, indicating that it\'s making the most of its resources without slowing down.
* ld71r18u44dws has a moderate CPU utilisation of 30.08%, which suggests that it\'s not heavily loaded but could still benefit from some optimization.

**Graphics Processing Unit (GPU) observations**

* ld71r16u13ws GPU average utilisation is relatively low at 39.03%, indicating underutilization.
* ld71r18u44cws GPU average utilisation is moderate at 48.01%.
* ld71r18u44fws GPU average utilisation is relatively high at 54.12%.

**Core observations**

* ld71r16u13ws core maximum utilisation is very high at 99.87%, indicating that the machine may be experiencing a bottleneck.
* ld71r18u44cws core maximum utilisation is moderate at 76.67%.
* ld71r18u44fws core average utilisation is relatively high at 83.92%.

That\'s all I have to report based on the provided data!''',

'''Based on the provided data, here are the central processing unit (CPU) average utilisations for each machine:

* ld71r18u44bws: 29.45%
* ld71r18u44dws: 30.08%
* ld71r18u44ews: 69.82%
* ld71r18u44cws: 57.09%
* ld71r16u15ws: 20.1% (Note: This machine has a null value for maximum utilisation, but the average utilisation is provided)
* ld71r16u13ws: Not applicable (since there\'s no data on CPU average utilisation)
* ld71r18u44fws: Not applicable (since there\'s no data on CPU average utilisation)
* ld71r16u14ws: Not applicable (since there\'s no data on CPU average utilisation)

Note that machines 6, 7, and 8 do not have available data for CPU average utilisations.'''],

                       ['''Based on the provided data, I found that the machine with the highest carbon emission value is:

ld71r16u14ws: 842.96

This result is based on the values extracted from the dictionary for each machine's "carbon emissions (gco2eq) - carbon emissions" value.''',
                        
                        ''' To find the machine with the highest carbon emission value, I'll need to look at the "carbon emissions (gco2eq) - carbon emissions" values for each machine.
After reviewing the data, I found that:
* ld71r16u14ws: 842.96
* ld71r18u44fws: 605.71
* ld71r18u44bws: 614.19
* ld71r18u44dws: 613.84
* ld71r18u44ews: 613.12
* ld71r16u15ws: 774.84
* ld71r16u13ws: 645.33
The machine with the highest carbon emission value is: ld71r16u14ws: 842.96''',

'''After reviewing the data, I found that:

* ld71r16u14ws: 842.96
* ld71r18u44ews: 613.12
* ld71r18u44dws: 613.84
* ld71r18u44fws: 605.71
* ld71r18u44bws: 614.19
* ld71r16u15ws: 774.84
* ld71r16u13ws: 645.33
* ld71r18u44cws: 598.92

The machine with the highest carbon emission value is:

ld71r16u15ws: 774.84''',

'''  To find the machine with the highest carbon emission value, I'll need to look at the "carbon emissions (gco2eq) - carbon emissions" values for each machine.

After reviewing the data, I found that:

* ld71r16u14ws: 842.96
* ld71r18u44bws: 614.19
* ld71r18u44cws: 598.92
* ld71r18u44dws: 613.84
* ld71r18u44ews: 613.12
* ld71r16u15ws: 774.84

The machine with the highest carbon emission value is:

ld71r16u15ws: 774.84''',

'''   Based on the provided data, the machine with the highest carbon emission value is:

ld71r16u14ws: 842.96''',

''' To find the machine with the highest carbon emission value, I'll need to look at the "carbon emissions (gco2eq) - carbon emissions" values for each machine.

According to the data, the machine with the highest carbon emission value is ld71r16u15ws: {'embodied carbon (gco2eq)': 418.9, 'carbon emissions (gco2eq) - carbon emissions': 774.84, ...}

This machine has a carbon emission value of 774.84 gCO2eq, which is the highest among all the machines.

So, the answer is: ld71r16u15ws''',

''' 
 Based on the provided data, the machine with the highest carbon emission value is still:

ld71r16u14ws: carbon emissions (gco2eq) - carbon emissions = **842.96**

This answer remains unchanged since there is no additional relevant data that contradicts or updates this conclusion.''',

'''Here are the "carbon emissions (gco2eq) - carbon emissions" values for each machine:

* ld71r16u13ws: 645.33
* ld71r16u14ws: 842.96
* ld71r16u15ws: 774.84
* ld71r18u44bws: 614.19
* ld71r18u44cws: 598.92
* ld71r18u44dws: 613.84
* ld71r18u44ews: 613.12
* ld71r18u44fws: 605.71

Comparing these values, I can see that the machine with the highest carbon emission value is:

ld71r16u14ws: 842.96''', 
''' To find the machine with the highest carbon emission value, I'll go through each dictionary in the list and extract the "carbon emissions (gco2eq) - carbon emissions" value. Then, I'll compare these values to determine 
which one is the highest.

Here are the "carbon emissions (gco2eq) - carbon emissions" values for each machine:

* ld71r16u13ws: 645.33
* ld71r18u44bws: 614.19
* ld71r18u44cws: 598.92
* ld71r18u44dws: 613.84
* ld71r18u44ews: 613.12
* ld71r16u14ws: 842.96
* ld71r16u15ws: 774.84

Comparing these values, I see that the machine with the highest carbon emission value is:

ld71r16u14ws: 842.96''',
'''
 Based on the provided data, I can see that the machine with the highest carbon emission value is:

ld71r16u14ws: 842.96

This result holds true even when considering additional relevant data such as embodied and operational carbon emissions, central processing unit maximum and average utilisation percentages, and core maximum and average utilisation percentages.

Therefore, my answer remains:

The machine with the highest carbon emission value is ld71r16u14ws, with a value of 842.96.'''],]
# Generate rows for each question and run
for i in range(len(base_data_set_1['question'])):
    for run in range(num_runs):
        rows.append({
            'question': base_data_set_1['question'][i],
            'ideal_response': base_data_set_1['ideal_response'][i],
            'provided_knowledge': base_data_set_1['provided_knowledge'][i],
            'run_number': run + 1,
            # 'llm_response': f"LLM response for question {i}, run {run+1}"  # Placeholder for actual LLM responses
            'llm_response': llm_responses_set_1[i][run]
        })

# Create the DataFrame
df_test_set = pd.DataFrame(rows)

print(df_test_set)

# Display the number of unique questions and total rows
print(f"\nNumber of unique questions: {df_test_set['question'].nunique()}")
print(f"Total number of rows: {len(df_test_set)}")


def evaluate_llm_responses(df: pd.DataFrame, evaluator_model: str = "mistral", temperature: float = 0.1) -> List[Dict]:
    """
    Evaluate LLM responses for correctness and faithfulness using the Mistral model from Ollama.
    
    Args:
    df (pd.DataFrame): DataFrame containing 'question', 'ideal_response', 'provided_knowledge', and 'llm_response' columns.
    evaluator_model (str): The model to use for evaluation (default is "mistral").
    
    Returns:
    List[Dict]: List of dictionaries containing evaluation results for each response.
    """
    results = []
    ollama_client = Client()  # Initialize Ollama client
    
    for _, group in df.groupby(['question', 'run_number']):
        row = group.iloc[0]
        prompt = f"""
        You are an expert evaluator of AI language model responses. Your task is to evaluate the following response for correctness and faithfulness.

        Question: {row['question']}
        
        Ideal Response: {row['ideal_response']}
        
        Provided Knowledge: {row['provided_knowledge']}
        
        LLM Response: {row['llm_response']}
        
        Please evaluate the LLM Response based on the following criteria:
        
        1. Correctness (0-10 scale): How well does the response align with the ideal response? The values must match.
        
        2. Faithfulness (0-10 scale): How well does the response adhere to the information provided in the 'Provided Knowledge'? It must not make up data.
        
        VERY IMPORTANT: YOUR SCORE MUST BE ONE OF THE FOLLOWING VALUES: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. INCLUDE NO PREAMBLE OR POST AMBLE
        Be specific in your explanation about what figures match between the llm response and the ideal response.
        Provide your evaluation in the following format:
        Correctness Score: [score]
        Correctness Explanation: [brief explanation]
        Faithfulness Score: [score]
        Faithfulness Explanation: [brief explanation]
        """
        
        response = ollama_client.generate(model=evaluator_model, prompt=prompt, options={"temperature": temperature})
        print(response)
        

        # Assuming the response is a dictionary with a 'response' key containing the generated text
        evaluation = response.get('response', '')
        print(evaluation)

        
        # Parse the evaluation
        eval_lines = evaluation.split('\n')
        correctness_score = None
        correctness_explanation = ''
        faithfulness_score = None
        faithfulness_explanation = ''
        
        for line in eval_lines:
            if line.startswith(' Correctness Score:'):
                correctness_score = float(line.split(':')[1].strip())
            elif line.startswith('Correctness Explanation:'):
                correctness_explanation = line.split(':')[1].strip()
            elif line.startswith('Faithfulness Score:'):
                faithfulness_score = float(line.split(':')[1].strip())
            elif line.startswith('Faithfulness Explanation:'):
                faithfulness_explanation = line.split(':')[1].strip()
        
        results.append({
            'question': row['question'],
            'run_number': row['run_number'],
            'correctness_score': correctness_score,
            'correctness_explanation': correctness_explanation,
            'faithfulness_score': faithfulness_score,
            'faithfulness_explanation': faithfulness_explanation
        })
  
    
   # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    
    # Merge the results with the original DataFrame
    merged_df = pd.merge(df, results_df, on=['question', 'run_number'], how='left')
    print(merged_df['correctness_score'])
    print(merged_df['correctness_explanation'])
    
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv('evaluation_results_with_original.csv', index=False)
    
    # Optionally, save to Excel
    # merged_df.to_excel('evaluation_results_with_original.xlsx', index=False)
    
    return merged_df

# Example usage
df_test_set = pd.DataFrame(rows)  # Assuming 'rows' is defined
evaluation_results_df = evaluate_llm_responses(df_test_set)

# Print the merged DataFrame
print(evaluation_results_df)