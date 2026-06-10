# Dictionaries

Dictionaries store key-value pairs.

```python
student = {
    "name": "Alice",
    "age": 25,
    "grade": "A"
}
```

## Accessing Values

```python
print(student["name"])     # Alice
print(student.get("age"))  # 25
print(student.get("city", "Not found"))  # Not found
```

## Useful Methods

```python
student = {"name": "Alice", "age": 25}

print(student.keys())    # dict_keys(['name', 'age'])
print(student.values())  # dict_values(['Alice', 25])
print(student.items())   # dict_items([('name', 'Alice'), ('age', 25)])

for key, value in student.items():
    print(f"{key}: {value}")
```

## Adding / Updating

```python
student["city"] = "New York"   # add new key
student["age"] = 26            # update existing key
```

## Dict Comprehensions

```python
squares = {x: x**2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```
