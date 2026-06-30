# Variables

Variables store data values. In JavaScript, you can declare variables with `var`, `let`, or `const`.

## Declaring Variables

```javascript
// Declare and assign separately
var a;
var b;
var sum;
a = 5;
b = 4;
sum = a + b;
console.log(sum);  // 9

// Declare and assign together
var a = 5;
var b = 4;
var sum = a + b;
console.log(sum);  // 9

// Declare multiple in one line
var a = 5, b = 4, sum = a + b;
console.log(sum);  // 9
```

## `var`, `let`, and `const`

| Keyword | Scope | Reassignable | Redeclarable | Hoisting |
|---------|-------|-------------|--------------|----------|
| `var` | Function-scoped | Yes | Yes | Hoisted (initialized as `undefined`) |
| `let` | Block-scoped | Yes | No | Hoisted (Temporal Dead Zone) |
| `const` | Block-scoped | **No** | No | Hoisted (Temporal Dead Zone) |

```javascript
var x = 1;        // function-scoped
let y = 2;        // block-scoped
const z = 3;      // block-scoped, can't be reassigned

z = 4;            // TypeError: Assignment to constant variable
```

## Variable Naming Rules

- Must start with a letter, `_`, or `$`
- Can contain letters, digits, `_`, and `$`
- Case-sensitive (`myVar` and `myvar` are different)
- Cannot use reserved keywords (`if`, `for`, `class`, etc.)

## Dynamic Typing

JavaScript is dynamically typed — a variable can hold any type of value:

```javascript
let value = 42;        // number
value = "hello";       // string
value = true;          // boolean
value = { a: 1 };      // object
value = null;          // null
value = undefined;     // undefined
```
