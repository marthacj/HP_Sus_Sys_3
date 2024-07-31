import pandas as pd
import numpy as np
import random
import string

# Create lists of different data types to test
num_items = 20

# Generate arrays
list1 = np.asarray([np.random.randint(0, 10) for n in range(num_items)])  # Integers
list2 = np.asarray([random.choice(string.ascii_letters) for n in range(num_items)])  # Strings
list3 = np.asarray([np.random.random() for n in range(num_items)])  # Floats
print(
    f"Array 1: {list1} \n"
    f"Shape: {np.shape(list1)} \n"
    f"Array 2: {list2} \n"
    f"Shape: {np.shape(list2)} \n"
    f"Array 3: {list3} \n"
    f"Shape: {np.shape(list3)} \n"
)

# Save as dict and convert to Pandas DataFrame
data = {
    "ints": list1,
    "strs": list2,
    "flts": list3,
}
dataframe = pd.DataFrame(data, columns=["ints", "strs", "flts",])
print(dataframe)