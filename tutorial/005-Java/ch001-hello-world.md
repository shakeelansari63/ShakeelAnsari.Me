# Hello, World!

Every Java program starts with a class definition and a `main` method — the entry point.

```java
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

## Structure Explained

| Part | Purpose |
|------|---------|
| `public class Main` | Defines a class named Main — the filename must match (`Main.java`) |
| `public static void main(String[] args)` | The JVM calls this method to start the program |
| `System.out.println()` | Prints text followed by a newline to the console |
| `String[] args` | Array of command-line arguments passed to the program |

## Compile and Run

```bash
javac Main.java          # compiles to Main.class
java Main                # runs the compiled bytecode
java Main arg1 arg2      # passes arguments to String[] args
```

Java source files are compiled to **bytecode** (`.class` files) which runs on the Java Virtual Machine (JVM).
