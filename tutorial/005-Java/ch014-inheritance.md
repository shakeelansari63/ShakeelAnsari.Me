# Inheritance

Inheritance allows a class to acquire the properties and methods of another class. It establishes an **is-a** relationship.

## Base and Derived Classes

```java
// Base class
public class Passenger {
    String name;
    String seatClass;

    public Passenger(String name) {
        this.name = name;
        this.seatClass = "Economy";
    }

    public void boardingAnnouncement() {
        System.out.println(name + " is boarding from Gate A");
    }

    public void printDetails() {
        System.out.println(name + " - " + seatClass);
    }
}

// Derived class
public class VipPassenger extends Passenger {
    String lounge;

    public VipPassenger(String name, String lounge) {
        super(name);  // call parent constructor
        this.seatClass = "Business";
        this.lounge = lounge;
    }

    @Override
    public void boardingAnnouncement() {
        System.out.println(name + " is boarding from VIP Gate");
    }

    public void accessLounge() {
        System.out.println(name + " accesses " + lounge + " lounge");
    }
}
```

## Using Inheritance

```java
public class Main {
    public static void main(String[] args) {
        Passenger p = new Passenger("Alice");
        VipPassenger vip = new VipPassenger("Bob", "Platinum");

        p.boardingAnnouncement();   // Alice is boarding from Gate A
        vip.boardingAnnouncement(); // Bob is boarding from VIP Gate
        vip.accessLounge();         // Bob accesses Platinum lounge
        vip.printDetails();         // Bob - Business
    }
}
```

## The `super` Keyword

- `super()` — calls the parent class constructor (must be the first statement)
- `super.methodName()` — calls the parent class method

## Method Overriding

A subclass can provide a specific implementation of a method from the parent class using `@Override`:

- Method must have the same signature (name + parameters)
- Access level cannot be more restrictive
- The `final` method in parent cannot be overridden

## The `protected` Access Modifier

| Modifier | Same Class | Same Package | Subclass (different package) | World |
|----------|------------|--------------|------------------------------|-------|
| `private` | Yes | No | No | No |
| default | Yes | Yes | No | No |
| `protected` | Yes | Yes | Yes | No |
| `public` | Yes | Yes | Yes | Yes |

Java doesn't support multiple inheritance of classes (a class can extend only one parent), but a class can implement multiple interfaces.
