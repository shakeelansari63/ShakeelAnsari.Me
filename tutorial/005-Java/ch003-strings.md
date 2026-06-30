# Strings

Strings in Java are objects, not primitives. They are immutable — any operation that seems to modify a string actually creates a new one.

## Creating Strings

```java
String greeting = "Hello";
String name = new String("World");  // explicit constructor (rarely used)
```

## Common String Methods

```java
String s = "Hello, Java!";

s.length();                    // 12
s.charAt(0);                   // 'H'
s.substring(7);                // "Java!"
s.substring(7, 11);            // "Java"
s.indexOf("Java");             // 7
s.contains("Java");            // true
s.startsWith("Hello");         // true
s.endsWith("!");               // true
s.toUpperCase();               // "HELLO, JAVA!"
s.toLowerCase();               // "hello, java!"
s.replace("Java", "World");    // "Hello, World!"
s.split(", ");                 // ["Hello", "Java!"]
s.trim();                      // removes leading/trailing whitespace
s.isEmpty();                   // false
```

## String Concatenation

```java
String first = "Hello";
String second = "World";
String result = first + ", " + second + "!";  // "Hello, World!"
```

For complex concatenation, prefer `StringBuilder`:

```java
StringBuilder sb = new StringBuilder();
sb.append("Hello");
sb.append(", ");
sb.append("World");
sb.append("!");
String result = sb.toString();
```

## String Comparison

Always use `.equals()` — never `==` for string equality:

```java
String a = "hello";
String b = "hello";
String c = new String("hello");

a == b;          // true (interned strings share the same reference)
a == c;          // false (different object)
a.equals(c);     // true (same characters)
```

## String Formatting

```java
String formatted = String.format("Name: %s, Age: %d, Score: %.1f", "Alice", 25, 92.5);
// "Name: Alice, Age: 25, Score: 92.5"
```
