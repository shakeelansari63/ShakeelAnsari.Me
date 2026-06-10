# Conditionals

Conditional statements let your code make decisions.

```python
age = 18

if age >= 18:
    print("You are an adult")
else:
    print("You are a minor")
```

## `elif`

```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

print(grade)  # B
```

## Comparison Operators

| Operator | Meaning             |
|----------|---------------------|
| `==`     | Equal to            |
| `!=`     | Not equal to        |
| `<`      | Less than           |
| `>`      | Greater than        |
| `<=`     | Less than or equal  |
| `>=`     | Greater than or equal |

## Logical Operators

```python
x = 5

print(x > 2 and x < 10)   # True
print(x > 2 or x < 0)     # True
print(not x > 2)           # False
```

## Truthiness

Values that are considered `False`: `0`, `""`, `[]`, `None`, `False`.

```python
if "":
    print("this won't run")

if "hello":
    print("this will run")
```
