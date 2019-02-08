# Tens-Files
This python module is used to save and load tensors in binary files. 

## Depedencies:
- Numpy (tensors)

## To save:
```python
import tens
tens.save(data, "data/tensor.tens")
```

## To load:
```python
import tens
data = tens.load("data/tensors.tens")
```

## Full example:
```python
# This example create, save then load a tensor

import tens
import numpy as np


# Constants
data_path = "data.tens"

width = 2
height = 3
depth = 4

# Create data
data = np.array([i for i in range(width * height * depth)])
data.shape = (depth, height, width)

print("Created data:", data)

# Save data
tens.save(data, data_path)

# Free data
data = None

# Load data
data = tens.load(data_path)
print("---")
print("Loaded data:", data)
```

## Other things:
- This project is under the MIT License
- I will maybe release a C++ version
