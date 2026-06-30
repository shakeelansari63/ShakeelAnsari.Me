# Immediately Invoked Function Expressions (IIFE)

An IIFE is a function that runs immediately after it's defined.

## Basic IIFE

```javascript
(function() {
    console.log("This runs immediately!");
})();
```

## With Parameters

```javascript
(function(name) {
    console.log("Hello, " + name + "!");
})("Alice");
```

## Arrow Function IIFE

```javascript
(() => {
    console.log("Arrow IIFE");
})();
```

## Creating Private Scope

IIFEs create their own scope, preventing variable leakage:

```javascript
(function() {
    let privateVar = "secret";
    console.log(privateVar);  // accessible here
})();

console.log(privateVar);  // ReferenceError: privateVar is not defined
```

## The Module Pattern

```javascript
const counter = (function() {
    let count = 0;  // private variable

    return {
        increment: function() { count++; },
        decrement: function() { count--; },
        getCount: function() { return count; }
    };
})();

counter.increment();
counter.increment();
console.log(counter.getCount());  // 2
console.log(counter.count);       // undefined (private)
```
