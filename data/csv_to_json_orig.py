import csv
import json
import sys

def csv_to_json(filename, as_json=True):
    data_dict = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for machine_row in reader:
            machine_key = machine_row[next(iter(machine_row))]
            data_dict[machine_key] = {}
            for header, value in machine_row.items():
                if header == next(iter(machine_row)):
                    """data_dict[machine_key]['carbon emission'] = value"""
                    continue
                """key before parentheses"""
                lower_level_key = header.split('(')[0].strip().replace('\n',' ')
                """key within parentheses"""
                top_level_key = header.split('(')[-1].split(')')[0].strip().replace('\n',' ')
                if top_level_key not in data_dict[machine_key]:
                    data_dict[machine_key][top_level_key] = {}
                try:
                    value = str(round(float(value),2)) 
                except ValueError:
                    value = value
                if top_level_key != lower_level_key:
                    data_dict[machine_key][top_level_key][lower_level_key] = value
                else:
                    data_dict[machine_key][top_level_key] = value
    if as_json:
        return json.dumps(data_dict)
    return data_dict
    

def flatten(data_dict):
    flat_dict = {}
    for machine_key, machine_data in data_dict.items():
        machine_key = machine_key[6:]
        for top_level_key, top_level_data in machine_data.items():
            if isinstance(top_level_data, dict):
                for lower_level_key, lower_level_data in top_level_data.items():
                    flat_dict[machine_key + '_' + top_level_key + '_' + lower_level_key] = lower_level_data
            else:
                flat_dict[machine_key + '_' + top_level_key] = top_level_data
    return flat_dict

def stringify(flat_dict):
    dict_string = ""
    for k,v in flat_dict.items():
        dict_string += k.replace('_',' ') + " = " + v + "\n"
    # Changed to dict_string return not flat_dict
    dict_string = deabbreviate(dict_string)
    print('dict_string:', dict_string)
    return dict_string

def deabbreviate(sentence: str) -> str:
    abbr_list = {' CPU ': ' central processing unit ', 'GPU': 'graphics processing unit', '%':'Percent ', 'Mem ':'Memory ', 'min ':'minimum', 'max ':'maximum ', 'avg ':'average ', '#': ' number of ', 'mb ':'megabytes ',
                 'oc ': 'occurrences '} 
    sentence = sentence.lower()
    for abbr, full in abbr_list.items():
        sentence = sentence.replace(abbr.lower(), full.lower())
    return sentence



if __name__ == '__main__':
    filename = r"C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\data\emissions1038-0610-0614-day.csv"
    filename = r"C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\data\1038-0610-0614-evening.xlsx"

    
    data_dict_json = csv_to_json(filename, as_json=False)
    print(flatten(data_dict_json))
    #  Added below
    flat_dict = flatten(data_dict_json)
    dict_string = stringify(flat_dict)
   
    # """save to file"""
    # with open('data.json', 'w') as f:
    #     f.write(dict_string)
    
    # print(dict_string)
 
    # """read it back in"""
    # with open('data.json', 'r') as f:
    #     data_dict = json.load(f)

    # for k,v in data_dict.items():
    #     print(data_dict[k]['carbon emissions'])
    #     print(data_dict[k]['CPU %Utilization']['#Cores'])
    #     print(k,v)

    #     input("Press Enter to continue...")


    # Write the stringified flat dictionary to a file
    with open('data.txt', 'w') as f:
        f.write(dict_string)

    # Read the stringified flat dictionary from the file
    with open('data.txt', 'r') as f:
        read_back_string = f.read()

    print("Stringified flat dictionary read back from the file:")
    print(read_back_string)
