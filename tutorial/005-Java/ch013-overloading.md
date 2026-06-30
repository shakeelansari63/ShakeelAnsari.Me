# Method Overloading

Method overloading allows multiple methods in the same class to share the same name with different parameters.

```java
public class Flight {
    int seats;
    int passengers;

    public Flight() {
        seats = 100;
    }

    public Flight(int seats) {
        this.seats = seats;
    }

    public Flight(int seats, int passengers) {
        this.seats = seats;
        this.passengers = passengers;
    }

    void addPassengers() {
        if (passengers < seats) passengers++;
    }

    void addPassengers(int count) {
        int available = seats - passengers;
        int toAdd = Math.min(count, available);
        passengers += toAdd;
    }

    void addPassengers(int count, boolean waitlisted) {
        int available = seats - passengers;
        if (count <= available) {
            passengers += count;
        } else if (waitlisted) {
            System.out.println(count + " passengers added to waitlist");
        }
    }
}
```

## Rules for Overloading

1. Methods must have the **same name**
2. Parameters must differ in **number**, **type**, or **order**
3. Return type **can** differ (but not as the sole differentiator)
4. Access modifier **can** differ

```java
// All valid overloads
int add(int a, int b);
double add(double a, double b);
int add(int a, int b, int c);
double add(int a, double b);
double add(double a, int b);
```

## Overloading Constructors

Constructors can also be overloaded:

```java
public class Point {
    int x, y;

    public Point() {
        this(0, 0);
    }

    public Point(int x) {
        this(x, 0);
    }

    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }
}
```

The compiler resolves which overload to call based on the arguments at compile time (static polymorphism).
