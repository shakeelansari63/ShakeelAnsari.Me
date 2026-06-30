# Command Line Arguments

The `main` method receives command-line arguments as a `String[]` array.

```java
public class Main {
    public static void main(String[] args) {
        for (int i = 0; i < args.length; i++) {
            System.out.println("Arg " + i + ": " + args[i]);
        }
    }
}
```

## Running with Arguments

```bash
javac Main.java
java Main Alice Bob Charlie
```

Output:
```
Arg 0: Alice
Arg 1: Bob
Arg 2: Charlie
```

## Parsing Numeric Arguments

Command-line arguments come as strings. Parse them to use as numbers:

```java
public class Main {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java Main <number1> <number2>");
            return;
        }

        int a = Integer.parseInt(args[0]);
        int b = Integer.parseInt(args[1]);
        System.out.println("Sum: " + (a + b));
    }
}
```

## Handling Parse Errors

```java
public static void main(String[] args) {
    try {
        int a = Integer.parseInt(args[0]);
        int b = Integer.parseInt(args[1]);
        System.out.println("Sum: " + (a + b));
    } catch (NumberFormatException e) {
        System.out.println("Invalid number format. Please provide integers.");
    } catch (ArrayIndexOutOfBoundsException e) {
        System.out.println("Please provide two numbers.");
    }
}
```
