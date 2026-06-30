# let and const

ES2015 introduced `let` and `const` as alternatives to `var` with better scoping rules.

## let vs var

```javascript
// var is function-scoped
var x = 10;
if (true) {
    var x = 20;  // same variable!
}
console.log(x);  // 20 — value was overwritten

// let is block-scoped
let y = 10;
if (true) {
    let y = 20;  // different variable (shadowing)
}
console.log(y);  // 10 — outer value preserved
```

## const

`const` declares a constant reference — the binding cannot be reassigned.

```javascript
const PI = 3.14159;
PI = 3;  // TypeError: Assignment to constant variable
```

## const with Objects and Arrays

`const` prevents reassignment but not mutation:

```javascript
const user = { name: "Alice" };
user.name = "Bob";     // allowed — mutating the object
user.age = 30;         // allowed — adding properties
user = {};             // TypeError — reassigning

const numbers = [1, 2, 3];
numbers.push(4);       // allowed — mutating the array
numbers = [];          // TypeError — reassigning
```

## Temporal Dead Zone (TDZ)

`let` and `const` are hoisted but not initialized — accessing them before declaration throws an error.

```javascript
console.log(x);  // undefined (var is hoisted and initialized)
var x = 5;

console.log(y);  // ReferenceError: Cannot access 'y' before initialization
let y = 5;
```

## Best Practice

- Use `const` by default
- Use `let` when you need to reassign
- Never use `var` in modern JavaScript
