# File Handling

Python can read and write files using the `open()` function.

## Opening and Reading a File

```python
file = open("test.txt", "r")
content = file.read()
print(content)
file.close()
```

Always close the file — or use `with` for automatic cleanup.

## The `with` Statement

```python
with open("test.txt", "r") as file:
    content = file.read()
    print(content)
# File is automatically closed here
```

## Reading Line by Line

```python
with open("test.txt", "r") as file:
    for line in file:
        print(line.strip())
```

## Writing to a File

```python
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("Second line\n")
```

Mode `"w"` overwrites. Use `"a"` to append.

```python
with open("output.txt", "a") as file:
    file.write("Appended line\n")
```

## File Modes

| Mode | Description          |
|------|----------------------|
| `"r"`| Read (default)        |
| `"w"`| Write (overwrites)    |
| `"a"`| Append               |
| `"r+"`| Read and write      |

## The `os` Module

```python
import os

print(os.getcwd())          # Current directory
os.listdir(".")              # List files in directory
os.path.exists("test.txt")  # True if file exists
os.rename("old.txt", "new.txt")
os.remove("test.txt")       # Delete a file
```
