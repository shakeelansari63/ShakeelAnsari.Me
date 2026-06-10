# Loops

Loops let you repeat code.

## `for` with `range()`

```python
for i in range(5):
    print(i)
# 0 1 2 3 4
```

`range(start, stop, step)`:

```python
for i in range(2, 10, 2):
    print(i)
# 2 4 6 8
```

## `for` over a List

```python
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

## `enumerate()`

Get both index and value:

```python
fruits = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits):
    print(i, fruit)
# 0 apple
# 1 banana
# 2 cherry
```

## `while` Loop

```python
count = 0
while count < 5:
    print(count)
    count += 1
```

## `break` and `continue`

```python
for i in range(10):
    if i == 3:
        continue  # skip this iteration
    if i == 7:
        break     # exit loop entirely
    print(i)
# 0 1 2 4 5 6
```
