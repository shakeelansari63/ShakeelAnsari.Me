# Return Values

Functions can return a value using the `return` keyword.

## Basic Return

```javascript
function square(x) {
    return x * x;
}

let result = square(5);  // 25
```

## Early Return

```javascript
function divide(a, b) {
    if (b === 0) {
        return "Cannot divide by zero";
    }
    return a / b;
}
```

## Returning Multiple Values

Use arrays or objects to return multiple values:

```javascript
// Array destructuring
function getCoordinates() {
    return [10, 20];
}
const [x, y] = getCoordinates();

// Object destructuring
function getUser() {
    return { name: "Alice", age: 30 };
}
const { name, age } = getUser();
```

## Implicit Return (Arrow Functions)

Arrow functions with a single expression automatically return the result:

```javascript
const square = x => x * x;
const add = (a, b) => a + b;
const getObj = () => ({ name: "Alice" });  // parentheses needed for object
```

## No Return

A function without a `return` statement returns `undefined`:

```javascript
function logMessage(msg) {
    console.log(msg);
    // no explicit return
}

let result = logMessage("Hello");  // undefined
```
