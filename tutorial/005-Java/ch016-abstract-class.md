# Abstract Classes and Interfaces

## Abstract Classes

An abstract class cannot be instantiated directly. It may contain abstract methods (without a body) that subclasses must implement.

```java
abstract class Animal {
    String name;

    public Animal(String name) {
        this.name = name;
    }

    // Concrete method
    public void eat() {
        System.out.println(name + " is eating");
    }

    // Abstract method — must be implemented by subclasses
    public abstract void makeSound();
}

class Dog extends Animal {
    public Dog(String name) {
        super(name);
    }

    @Override
    public void makeSound() {
        System.out.println("Woof!");
    }
}

class Cat extends Animal {
    public Cat(String name) {
        super(name);
    }

    @Override
    public void makeSound() {
        System.out.println("Meow!");
    }
}
```

## Interfaces

An interface defines a contract — a set of method signatures that implementing classes must provide.

```java
interface Digger {
    void dig();
}

interface Protector {
    void protect();
}

interface Scratcher {
    void scratch();
}
```

## Implementing Interfaces

```java
class Dog extends Animal implements Digger, Protector {
    public Dog(String name) {
        super(name);
    }

    public void makeSound() { System.out.println("Woof!"); }
    public void dig() { System.out.println(name + " is digging"); }
    public void protect() { System.out.println(name + " is guarding"); }
}

class Cat extends Animal implements Scratcher {
    public Cat(String name) {
        super(name);
    }

    public void makeSound() { System.out.println("Meow!"); }
    public void scratch() { System.out.println(name + " is scratching"); }
}
```

## Abstract Class vs Interface

| Feature | Abstract Class | Interface |
|---------|---------------|-----------|
| Instantiation | Cannot be instantiated | Cannot be instantiated |
| Methods | Can have both abstract and concrete methods | Methods are abstract by default (Java 8+ allows `default` and `static` methods) |
| Fields | Can have any fields (instance, static, final) | Only `public static final` constants |
| Constructors | Can have constructors | Cannot have constructors |
| Inheritance | A class can extend only one abstract class | A class can implement multiple interfaces |
| `extends` vs `implements` | `extends` | `implements` |

## When to Use Which

- **Abstract class** — when classes share state (fields) or partial implementation
- **Interface** — when unrelated classes need to share behavior (a capability contract)
