# Classes and Objects

Classes are blueprints for objects. Objects are instances of classes.

## Defining a Class

```java
class Flight {
    int seats = 100;
    int passengers;
    static int totalFlights;
    static int totalPassengers;

    public Flight() {
        totalFlights++;
    }

    public Flight(int seats) {
        this();
        this.seats = seats;
    }

    public Flight(int seats, int passengers) {
        this(seats);
        this.passengers = passengers;
        totalPassengers += passengers;
    }

    void bookSeat() {
        if (passengers < seats) {
            passengers++;
            totalPassengers++;
        }
    }

    void cancelBooking() {
        if (passengers > 0) {
            passengers--;
            totalPassengers--;
        }
    }

    void printStatus() {
        System.out.println("Seats available: " + (seats - passengers));
        System.out.println("Passengers: " + passengers);
    }

    static void printTotalInfo() {
        System.out.println("Total flights: " + totalFlights);
        if (totalFlights > 0) {
            System.out.println("Average passengers per flight: " +
                (float) totalPassengers / totalFlights);
        }
    }
}
```

## Using the Class

```java
public class Main {
    public static void main(String[] args) {
        Flight flight1 = new Flight();
        Flight flight2 = new Flight(200);
        Flight flight3 = new Flight(150, 50);

        for (int i = 0; i < 10; i++) {
            flight1.bookSeat();
        }

        flight1.printStatus();
        Flight.printTotalInfo();
    }
}
```

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Constructor** | Special method called when creating an object with `new` |
| `this()` | Calls another constructor in the same class (constructor chaining) |
| `this` | Reference to the current object instance |
| **Static members** | Belong to the class, not instances — shared across all objects |
| **Instance members** | Belong to each object — each has its own copy |
