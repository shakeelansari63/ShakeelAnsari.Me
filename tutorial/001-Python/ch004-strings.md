# Strings

Strings are sequences of characters enclosed in quotes.

```python
single = 'hello'
double = "hello"
```

## Concatenation

```python
first = "Hello"
last = "World"
result = first + " " + last
print(result)  # Hello World
```

## f-Strings (Python 3.6+)

```python
name = "Alice"
age = 25
print(f"My name is {name} and I am {age} years old")
```

## `.format()` Method

```python
print("My name is {} and I am {} years old".format("Alice", 25))
```

## Common String Methods

```python
text = "  Hello, World!  "

print(text.upper())        # "  HELLO, WORLD!  "
print(text.lower())        # "  hello, world!  "
print(text.strip())        # "Hello, World!"
print(text.replace("World", "Python"))  # "  Hello, Python!  "
print(text.split(","))     # ['  Hello', ' World!  ']
print(":".join(["a", "b", "c"]))  # "a:b:c"
```

## Slicing

```python
s = "Python"
print(s[0])    # P
print(s[-1])   # n
print(s[0:3])  # Pyt
print(s[:3])   # Pyt
print(s[3:])   # hon
print(s[::2])  # Pto
```
