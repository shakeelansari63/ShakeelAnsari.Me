# Variables & Data Types

Variables store data. Python is dynamically typed — you don't need to declare a type.

```python
name = "Alice"
age = 25
height = 5.6
is_student = True

print(name, age, height, is_student)
```

## Basic Data Types

| Type    | Example          | Description        |
|---------|------------------|--------------------|
| `int`   | `42`, `-3`, `0`  | Whole numbers      |
| `float` | `3.14`, `-0.5`   | Decimal numbers    |
| `str`   | `"hello"`, `'hi'`| Text               |
| `bool`  | `True`, `False`  | Boolean values     |

## The `type()` Function

Use `type()` to check a variable's type:

```python
print(type(42))       # <class 'int'>
print(type(3.14))     # <class 'float'>
print(type("hello"))  # <class 'str'>
print(type(True))     # <class 'bool'>
```

## Dynamic Typing

A variable can change type:

```python
x = 10
print(type(x))  # int
x = "now a string"
print(type(x))  # str
```

## Naming Conventions

- Use lowercase with underscores (`snake_case`): `my_variable`, `user_name`
- Names are case-sensitive: `age` and `Age` are different
- Cannot start with a digit: `1var` is invalid
- Avoid reserved keywords: `if`, `for`, `while`, `class`, etc.
