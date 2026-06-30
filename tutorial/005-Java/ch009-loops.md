# Loops

Java provides three loop constructs: `for`, `while`, and `do-while`.

## For Loop

```java
for (int i = 0; i < 5; i++) {
    System.out.println("Iteration: " + i);
}
```

Components:
1. **Initialization** — runs once at the start (`int i = 0`)
2. **Condition** — checked before each iteration (`i < 5`)
3. **Update** — runs after each iteration (`i++`)

## For-Each Loop

Iterates over arrays and collections without an index:

```java
int[] numbers = {10, 20, 30, 40, 50};
for (int num : numbers) {
    System.out.println(num);
}
```

## While Loop

```java
int i = 0;
while (i < 5) {
    System.out.println("Count: " + i);
    i++;
}
```

The condition is checked before the body executes — the body may never run.

## Do-While Loop

```java
int i = 0;
do {
    System.out.println("Count: " + i);
    i++;
} while (i < 5);
```

The body runs at least once — the condition is checked after.

## Nested Loops

```java
for (int i = 1; i <= 3; i++) {
    for (int j = 1; j <= 3; j++) {
        System.out.print(i * j + " ");
    }
    System.out.println();
}
```

## Break and Continue

```java
// Break exits the loop
for (int i = 0; i < 10; i++) {
    if (i == 5) break;    // exits loop when i is 5
    System.out.print(i);   // prints: 01234
}

// Continue skips to next iteration
for (int i = 0; i < 10; i++) {
    if (i % 2 == 0) continue;  // skip even numbers
    System.out.print(i);        // prints: 13579
}
```

## Labeled Break and Continue

```java
outer:
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        if (i == 1 && j == 1) break outer;  // exits both loops
        System.out.println(i + "," + j);
    }
}
```
