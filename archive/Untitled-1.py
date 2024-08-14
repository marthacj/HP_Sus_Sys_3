
def calculation(param):
    total_emissions = 0
    for machine in param:
        total_emissions += machine["carbon emissions (gco2eq)"]
    return total_emissions

param = eval(''' [
    {
        "machine": "ld71r18u44bws",
        "carbon emissions (gco2eq)": 514.77,
        "embodied carbon (gco2eq)": 506.69,
        "operational carbon (gco2eq)": 8.08
    },
    {
        "machine": "ld71r18u44fws",
        "carbon emissions (gco2eq)": 517.46,
        "embodied carbon (gco2eq)": 506.69,
        "operational carbon (gco2eq)": 10.77
    },
    {
        "machine": "ld71r16u13ws",
        "carbon emissions (gco2eq)": 436.22,
        "embodied carbon (gco2eq)": 418.89,
        "operational carbon (gco2eq)": 17.33
    },
    {
        "machine": "ld71r18u44cws",
        "carbon emissions (gco2eq)": 514.53,
        "embodied carbon (gco2eq)": 506.69,
        "operational carbon (gco2eq)": 7.83
    },
    {
        "machine": "ld71r16u15ws",
        "carbon emissions (gco2eq)": 435.24,
        "embodied carbon (gco2eq)": 418.89,
        "operational carbon (gco2eq)": 16.35
    },
    {
        "machine": "ld71r16u14ws",
        "carbon emissions (gco2eq)": 432.24,
        "embodied carbon (gco2eq)": 418.89,
        "operational carbon (gco2eq)": 13.35
    },
    {
        "machine": "ld71r18u44dws",
        "carbon emissions (gco2eq)": 513.44,
        "embodied carbon (gco2eq)": 506.69,
        "operational carbon (gco2eq)": 6.75
    },
    {
        "machine": "ld71r18u44ews",
        "carbon emissions (gco2eq)": 515.85,
        "embodied carbon (gco2eq)": 506.69
    }
]''')
print(calculation(param))




