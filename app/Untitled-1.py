def calculation(param):
    for machine in param:
        cpu_over_80 = machine['central_processing_unit_total_seconds_over_80percent']
        if cpu_over_80 > max(param, key=lambda x: x['cpu_over_80'])['cpu_over_80']:
            return machine['machine']

def calculation(param):
    for machine in param:
        cpu_over_80 = machine['central_processing_unit_total_seconds_over_80percent']
        if cpu_over_80 > max(param, key=lambda x: x['cpu_over_80'])['cpu_over_80']:
            return machine['machine']

param = eval(''' [
    {
        "machine": "ld71r18u44bws",
        "central_processing_unit_max_utilization": 99.716,
        "central_processing_unit_avg_utilization": 76.988,
        "central_processing_unit_total_seconds_over_80percent": 480,
        "central_processing_unit_number_of_occurrences_over_80percent": 4,
        "number_of_cores": 24,
        "graphics_processing_unit_max_utilization": 95
    }
]''')
print(calculation(param))