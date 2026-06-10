# Functions

Functions are reusable blocks of code.

```python
def greet():
    print("Hello!")

greet()  # Hello!
```

## Parameters & Return

```python
def add(a, b):
    return a + b

result = add(3, 5)
print(result)  # 8
```

## Default Arguments

```python
def greet(name="World"):
    print(f"Hello, {name}!")

greet()          # Hello, World!
greet("Alice")   # Hello, Alice!
```

## `*args` and `**kwargs`

```python
def add_all(*args):
    return sum(args)

print(add_all(1, 2, 3, 4))  # 10


def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25)
```

## Lambda Functions

One-line anonymous functions:

```python
square = lambda x: x ** 2
print(square(5))  # 25

numbers = [1, 2, 3, 4]
doubled = list(map(lambda x: x * 2, numbers))
print(doubled)  # [2, 4, 6, 8]
```
