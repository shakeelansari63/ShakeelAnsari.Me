# Methods

Methods define reusable blocks of code in Java.

## Defining a Method

```java
public static int add(int a, int b) {
    return a + b;
}
```

| Part | Meaning |
|------|---------|
| `public` | Access modifier (visible to all) |
| `static` | Belongs to the class, not an instance |
| `int` | Return type (use `void` for no return) |
| `add` | Method name |
| `int a, int b` | Parameters |

## Calling a Method

```java
int result = add(5, 3);   // result = 8
```

## Method Overloading

Multiple methods can have the same name with different parameters:

```java
public static int add(int a, int b) {
    return a + b;
}

public static double add(double a, double b) {
    return a + b;
}

public static int add(int a, int b, int c) {
    return a + b + c;
}
```

The compiler picks the correct overload based on the arguments provided.

## Pass by Value

Java passes all arguments **by value**. For primitives, the method gets a copy. For objects, the method gets a copy of the reference.

```java
public static void changeValue(int x) {
    x = 100;  // only changes the local copy
}

int num = 5;
changeValue(num);
System.out.println(num);  // still 5
```

## Varargs (Variable Arguments)

```java
public static int sum(int... numbers) {
    int total = 0;
    for (int n : numbers) {
        total += n;
    }
    return total;
}

sum(1, 2);           // 3
sum(1, 2, 3, 4, 5); // 15
sum();               // 0
```

## Recursion

```java
public static int factorial(int n) {
    if (n <= 1) return 1;        // base case
    return n * factorial(n - 1);  // recursive call
}
```
