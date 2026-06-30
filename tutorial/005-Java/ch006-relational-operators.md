# Relational Operators

Relational operators compare two values and return a boolean result.

## Comparison Operators

```java
int a = 10, b = 20;

a == b;   // false — equal to
a != b;   // true  — not equal to
a > b;    // false — greater than
a < b;    // true  — less than
a >= b;   // false — greater than or equal to
a <= b;   // true  — less than or equal to
```

## Using in Conditions

```java
int age = 18;

if (age >= 18) {
    System.out.println("You are an adult");
}

int score = 85;
char grade = (score >= 90) ? 'A' : (score >= 80) ? 'B' : 'C';
```

## Comparing Objects

For objects, `==` compares **references** (memory addresses), not **values**:

```java
String s1 = new String("hello");
String s2 = new String("hello");

s1 == s2;         // false — different objects
s1.equals(s2);    // true  — same characters
```

Always use `.equals()` for string comparison and implement `equals()` in your own classes.

## Floating-Point Comparison

Floating-point arithmetic has precision issues. Compare with a tolerance (epsilon):

```java
double a = 0.1 + 0.2;   // 0.30000000000000004
double b = 0.3;

a == b;                  // false!

double epsilon = 0.000001;
Math.abs(a - b) < epsilon;  // true
```
