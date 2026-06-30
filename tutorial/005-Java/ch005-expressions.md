# Expressions and Operators

Operators in Java are used to perform operations on variables and values.

## Arithmetic Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `5 + 3` → 8 |
| `-` | Subtraction | `5 - 3` → 2 |
| `*` | Multiplication | `5 * 3` → 15 |
| `/` | Division | `5 / 2` → 2 (integer) |
| `%` | Modulus (remainder) | `5 % 2` → 1 |

Integer division truncates toward zero. Use `double` for fractional results.

```java
int a = 5 / 2;      // 2
double b = 5.0 / 2; // 2.5
```

## Compound Assignment Operators

```java
int x = 10;
x += 5;   // x = 15
x -= 3;   // x = 12
x *= 2;   // x = 24
x /= 4;   // x = 6
x %= 3;   // x = 0
```

## Increment and Decrement

```java
int i = 5;
i++;      // post-increment: i becomes 6
++i;      // pre-increment: i becomes 7
i--;      // post-decrement
--i;      // pre-decrement

int x = 5;
int y = x++;  // y = 5, x = 6 (post: assign then increment)
int z = ++x;  // z = 7, x = 7 (pre: increment then assign)
```

## Relational Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal to | `5 == 5` → true |
| `!=` | Not equal | `5 != 3` → true |
| `>` | Greater than | `5 > 3` → true |
| `<` | Less than | `5 < 3` → false |
| `>=` | Greater or equal | `5 >= 5` → true |
| `<=` | Less or equal | `5 <= 3` → false |

## Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&&` | AND (short-circuit) | `(5 > 3) && (2 < 4)` → true |
| `||` | OR (short-circuit) | `(5 > 3) || (2 > 4)` → true |
| `!` | NOT | `!(5 > 3)` → false |
