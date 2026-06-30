# User Input

Java provides several ways to read user input. The most common is the `Scanner` class.

## Using Scanner

```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter your name: ");
        String name = scanner.nextLine();

        System.out.print("Enter your age: ");
        int age = scanner.nextInt();

        System.out.println("Hello, " + name + "! You are " + age + " years old.");

        scanner.close();
    }
}
```

## Scanner Methods

| Method | Reads |
|--------|-------|
| `nextLine()` | A full line of text |
| `next()` | A single token (word) |
| `nextInt()` | An integer |
| `nextDouble()` | A double |
| `nextBoolean()` | A boolean |
| `nextByte()` | A byte |

## Handling Input Mismatch

```java
import java.util.InputMismatchException;

Scanner scanner = new Scanner(System.in);
System.out.print("Enter a number: ");

try {
    int number = scanner.nextInt();
    System.out.println("You entered: " + number);
} catch (InputMismatchException e) {
    System.out.println("Invalid input! Please enter a valid integer.");
}
```

## Reading from Console (Java 6+)

```java
import java.io.Console;

Console console = System.console();
if (console != null) {
    String username = console.readLine("Username: ");
    char[] password = console.readPassword("Password: ");
    System.out.println("Welcome, " + username);
}
```

`System.console()` returns `null` in non-interactive environments (like IDEs).

## Parsing Command-Line Arguments

```java
public class Main {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java Main <name> <age>");
            return;
        }
        String name = args[0];
        int age = Integer.parseInt(args[1]);
        System.out.println(name + " is " + age + " years old.");
    }
}
```
