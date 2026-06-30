# Variables and Constants

Java is statically typed — every variable must declare its type before use.

## Primitive Data Types

| Type | Size | Range |
|------|------|-------|
| `byte` | 8 bits | -128 to 127 |
| `short` | 16 bits | -32,768 to 32,767 |
| `int` | 32 bits | -2^31 to 2^31-1 |
| `long` | 64 bits | -2^63 to 2^63-1 |
| `float` | 32 bits | ~±3.4e38 (6-7 decimal digits) |
| `double` | 64 bits | ~±1.8e308 (15 decimal digits) |
| `char` | 16 bits | Unicode character (0 to 65,535) |
| `boolean` | JVM-dependent | `true` or `false` |

## Declaring Variables

```java
int a;
int b;
int sum;
a = 5;
b = 4;
sum = a + b;
```

Or declare and assign together:

```java
int a = 5;
int b = 4;
int sum = a + b;
```

Or multiple variables in one line:

```java
int a = 5, b = 4, sum = a + b;
```

## Constants with `final`

The `final` keyword makes a variable immutable:

```java
final int MAX_USERS = 100;
final double PI = 3.14159;
```

Attempting to reassign a `final` variable causes a compilation error.

## Default Values

Local variables **must** be initialized before use. Class/instance variables get defaults:

| Type | Default |
|------|---------|
| `int`, `long`, etc. | 0 |
| `double`, `float` | 0.0 |
| `boolean` | false |
| Object references | `null` |
