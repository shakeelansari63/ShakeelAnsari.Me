# Arithmetic Operators

```javascript
let a = 10, b = 3;

a + b;   // 13  — addition
a - b;   // 7   — subtraction
a * b;   // 30  — multiplication
a / b;   // 3.3333 — division
a % b;   // 1   — modulus (remainder)
a ** b;  // 1000 — exponentiation (ES2016)
```

## Operator Precedence

```javascript
2 + 3 * 4;     // 14 (multiplication first)
(2 + 3) * 4;   // 20 (parentheses override)
```

## Increment and Decrement

```javascript
let x = 5;
x++;      // post-increment: returns 5, then x = 6
++x;      // pre-increment: x = 7, returns 7
x--;      // post-decrement: returns 7, then x = 6
--x;      // pre-decrement: x = 5, returns 5
```

## Compound Assignment

```javascript
let x = 10;
x += 5;   // x = 15
x -= 3;   // x = 12
x *= 2;   // x = 24
x /= 4;   // x = 6
x %= 5;   // x = 1
x **= 3;  // x = 1
```

## NaN and Infinity

```javascript
"hello" * 5;   // NaN (Not a Number)
1 / 0;         // Infinity
-1 / 0;        // -Infinity

isNaN(NaN);    // true
isFinite(Infinity); // false
```
