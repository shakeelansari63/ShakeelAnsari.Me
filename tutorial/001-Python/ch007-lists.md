# Lists & Tuples

## Lists

Lists are ordered, mutable collections.

```python
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
```

### Indexing & Slicing

```python
fruits = ["apple", "banana", "cherry", "date"]
print(fruits[0])    # apple
print(fruits[-1])   # date
print(fruits[1:3])  # ['banana', 'cherry']
```

### Common Methods

```python
fruits = ["apple", "banana"]

fruits.append("cherry")     # ['apple', 'banana', 'cherry']
fruits.insert(1, "blueberry")  # ['apple', 'blueberry', 'banana', 'cherry']
fruits.pop()                # removes and returns last item
fruits.remove("banana")     # removes first match
fruits.sort()               # sorts in place
print(len(fruits))          # length
```

## Tuples

Tuples are ordered, **immutable** collections.

```python
point = (3, 4)
print(point[0])   # 3
```

### Lists vs Tuples

| Lists               | Tuples              |
|---------------------|---------------------|
| Mutable             | Immutable           |
| `[1, 2, 3]`         | `(1, 2, 3)`        |
| Use when data changes | Use for fixed data |
