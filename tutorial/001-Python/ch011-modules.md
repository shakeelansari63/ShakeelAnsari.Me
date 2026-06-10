# Modules & Packages

Modules let you organize code into separate files. Import them with `import`.

```python
# math is a built-in module
import math
print(math.sqrt(16))  # 4.0
```

## Import Specific Parts

```python
from math import sqrt, pi
print(sqrt(25))  # 5.0
print(pi)        # 3.141592653589793
```

## Import with Alias

```python
import numpy as np
import pandas as pd
```

## Your Own Module

Save `mymod.py`:

```python
def greet(name):
    return f"Hello, {name}!"
```

Use it in another file:

```python
import mymod
print(mymod.greet("Alice"))  # Hello, Alice!
```

## `__name__` Variable

`__name__` is `"__main__"` when the file is run directly.

```python
# mymod.py
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))  # Only runs when executed directly
```

## Packages

A package is a folder with an `__init__.py` file.

```
mypackage/
    __init__.py
    utils.py
    models.py
```

```python
from mypackage import utils
```

## Installing Packages with pip

```bash
pip install requests
pip install numpy pandas matplotlib
```

List installed packages:

```bash
pip list
```
