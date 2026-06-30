# Polymorphism

Polymorphism means "many forms" — a single method call can behave differently depending on the actual object type at runtime.

## Runtime Polymorphism

```java
public class Passenger {
    String name;

    public Passenger(String name) {
        this.name = name;
    }

    public void printDetails() {
        System.out.println("Passenger: " + name);
    }
}

public class VipPassenger extends Passenger {
    public VipPassenger(String name) {
        super(name);
    }

    @Override
    public void printDetails() {
        System.out.println("VIP Passenger: " + name + " (Priority Boarding)");
    }
}
```

## Polymorphic Behavior

```java
public class Main {
    public static void main(String[] args) {
        Passenger p1 = new Passenger("Alice");
        Passenger p2 = new VipPassenger("Bob");

        p1.printDetails();   // Passenger: Alice
        p2.printDetails();   // VIP Passenger: Bob (Priority Boarding)

        // Array of different types
        Passenger[] passengers = {
            new Passenger("Charlie"),
            new VipPassenger("Diana"),
            new Passenger("Eve")
        };

        for (Passenger p : passengers) {
            p.printDetails();  // runs the correct override for each
        }
    }
}
```

## The `instanceof` Operator

Check the actual type of an object at runtime:

```java
if (p2 instanceof VipPassenger) {
    VipPassenger vip = (VipPassenger) p2;  // downcast
    vip.accessLounge();
}
```

## Polymorphism with Interfaces

```java
interface Animal {
    void makeSound();
}

class Dog implements Animal {
    public void makeSound() {
        System.out.println("Woof!");
    }
}

class Cat implements Animal {
    public void makeSound() {
        System.out.println("Meow!");
    }
}

// Polymorphic call
Animal a1 = new Dog();
Animal a2 = new Cat();
a1.makeSound();  // Woof!
a2.makeSound();  // Meow!
```

Polymorphism enables code to work with objects of different types through a common interface or superclass, making the system extensible and maintainable.
