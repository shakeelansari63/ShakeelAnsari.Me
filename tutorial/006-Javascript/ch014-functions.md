# Functions

Functions are reusable blocks of code. They are first-class objects in JavaScript.

## Function Declaration

```javascript
function greet(name) {
    return "Hello, " + name + "!";
}

console.log(greet("Alice"));  // Hello, Alice!
```

## Function Expression

```javascript
const greet = function(name) {
    return "Hello, " + name + "!";
};
```

## Arrow Functions (ES6)

```javascript
const greet = (name) => {
    return "Hello, " + name + "!";
};

// Shorthand for single expression
const greet = name => "Hello, " + name + "!";

// Multiple parameters need parentheses
const add = (a, b) => a + b;

// No parameters
const hello = () => console.log("Hello!");
```

## Default Parameters

```javascript
function greet(name = "Guest") {
    return "Hello, " + name + "!";
}

greet();          // Hello, Guest!
greet("Alice");   // Hello, Alice!
```

## Rest Parameters

```javascript
function sum(...numbers) {
    return numbers.reduce((total, n) => total + n, 0);
}

sum(1, 2, 3);        // 6
sum(4, 5, 6, 7, 8);  // 30
```

## Function as First-Class Citizen

```javascript
// Pass function as argument
function operate(a, b, operation) {
    return operation(a, b);
}

operate(5, 3, (x, y) => x + y);  // 8
operate(5, 3, (x, y) => x * y);  // 15

// Return function
function createMultiplier(multiplier) {
    return (n) => n * multiplier;
}

const double = createMultiplier(2);
double(5);  // 10
```
