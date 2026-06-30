# Control Flow

Java provides standard control flow statements: `if`, `else if`, `else`, and `switch`.

## If-Else

```java
int score = 85;

if (score >= 90) {
    System.out.println("Grade: A");
} else if (score >= 80) {
    System.out.println("Grade: B");
} else if (score >= 70) {
    System.out.println("Grade: C");
} else if (score >= 60) {
    System.out.println("Grade: D");
} else {
    System.out.println("Grade: F");
}
```

## Ternary Operator

A shorthand for simple if-else:

```java
int age = 20;
String status = (age >= 18) ? "Adult" : "Minor";
```

## Switch Statement

```java
int day = 3;
String dayName;

switch (day) {
    case 1:
        dayName = "Monday";
        break;
    case 2:
        dayName = "Tuesday";
        break;
    case 3:
        dayName = "Wednesday";
        break;
    case 4:
        dayName = "Thursday";
        break;
    case 5:
        dayName = "Friday";
        break;
    case 6:
    case 7:
        dayName = "Weekend";
        break;
    default:
        dayName = "Invalid day";
}
```

Without `break`, execution falls through to the next case (fall-through).

## Enhanced Switch (Java 14+)

```java
String dayName = switch (day) {
    case 1 -> "Monday";
    case 2 -> "Tuesday";
    case 3 -> "Wednesday";
    case 4 -> "Thursday";
    case 5 -> "Friday";
    case 6, 7 -> "Weekend";
    default -> "Invalid day";
};
```

The arrow syntax (`->`) has implicit break and can also be used as an expression returning a value.

## Switch with Yielding (Java 14+)

```java
String result = switch (score) {
    case 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100 -> "A";
    case 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 -> "B";
    default -> {
        if (score >= 70) yield "C";
        else if (score >= 60) yield "D";
        else yield "F";
    }
};
```
