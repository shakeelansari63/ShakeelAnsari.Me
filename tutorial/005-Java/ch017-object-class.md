# Object, Wrapper Classes, Enums, and Records

## The Object Class

Every Java class implicitly extends `java.lang.Object`. Key methods to override:

```java
// OldFlight implementing toString and equals manually
public class OldFlight {
    String flightNumber;
    int seats;

    @Override
    public String toString() {
        return "Flight " + flightNumber + " (" + seats + " seats)";
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        OldFlight that = (OldFlight) obj;
        return seats == that.seats &&
               Objects.equals(flightNumber, that.flightNumber);
    }

    @Override
    public int hashCode() {
        return Objects.hash(flightNumber, seats);
    }
}

// NewFlight using Java 14+ features
public class NewFlight {
    String flightNumber;
    int seats;

    public NewFlight(String flightNumber, int seats) {
        this.flightNumber = flightNumber;
        this.seats = seats;
    }

    @Override
    public String toString() {
        return String.format("NewFlight[flightNumber=%s, seats=%d]", flightNumber, seats);
    }
}
```

## Wrapper Classes

Each primitive type has a corresponding wrapper class:

| Primitive | Wrapper |
|-----------|---------|
| `int` | `Integer` |
| `double` | `Double` |
| `boolean` | `Boolean` |
| `char` | `Character` |
| `long` | `Long` |
| `float` | `Float` |
| `byte` | `Byte` |
| `short` | `Short` |

```java
int primitive = 42;
Integer wrapper = primitive;           // autoboxing
int back = wrapper;                    // unboxing

Integer parsed = Integer.parseInt("123");
String hex = Integer.toHexString(255); // "ff"
boolean cmp = Integer.valueOf(5).equals(Integer.valueOf(5)); // true
```

## Enums

Enums define a fixed set of named constants.

```java
public enum SimpleFlightCrew {
    CAPTAIN,
    FIRST_OFFICER,
    FLIGHT_ATTENDANT
}
```

Enums can have fields, methods, and constructors:

```java
public enum AdvancedFlightCrew {
    CAPTAIN("Captain", 100000),
    FIRST_OFFICER("First Officer", 75000),
    FLIGHT_ATTENDANT("Flight Attendant", 45000);

    private final String title;
    private final int salary;

    AdvancedFlightCrew(String title, int salary) {
        this.title = title;
        this.salary = salary;
    }

    public String getTitle() { return title; }
    public int getSalary() { return salary; }
}

// Usage
AdvancedFlightCrew crew = AdvancedFlightCrew.CAPTAIN;
System.out.println(crew.getTitle());   // Captain
System.out.println(crew.getSalary());  // 100000
```

## Records (Java 16+)

Records are concise data carriers — they auto-generate constructors, getters, `equals()`, `hashCode()`, and `toString()`.

```java
// Traditional class
public class PassengerClass {
    private final String name;
    private final String seatClass;

    public PassengerClass(String name, String seatClass) {
        this.name = name;
        this.seatClass = seatClass;
    }

    public String name() { return name; }
    public String seatClass() { return seatClass; }
}

// Record — same functionality in one line
public record PassengerRecord(String name, String seatClass) {}
```

Records are immutable, transparent, and perfect for data transfer objects (DTOs).
