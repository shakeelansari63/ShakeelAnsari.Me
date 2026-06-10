# Classes & Objects

Classes define blueprints for objects.

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says woof!"

my_dog = Dog("Rex", 3)
print(my_dog.name)   # Rex
print(my_dog.bark()) # Rex says woof!
```

## `__init__` and `self`

- `__init__` is the constructor, called when creating an object.
- `self` refers to the current instance.

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

rect = Rectangle(4, 5)
print(rect.area())  # 20
```

## Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Cat(Animal):
    def speak(self):
        return f"{self.name} says meow!"

class Dog(Animal):
    def speak(self):
        return f"{self.name} says woof!"

animals = [Cat("Whiskers"), Dog("Rex")]
for a in animals:
    print(a.speak())
# Whiskers says meow!
# Rex says woof!
```
