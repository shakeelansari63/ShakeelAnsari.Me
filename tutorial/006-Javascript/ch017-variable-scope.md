# Variable Scope

Scope determines where variables are accessible in your code.

## Global Scope

Variables declared outside any function are global — accessible everywhere.

```javascript
let globalVar = "I'm global";

function test() {
    console.log(globalVar);  // accessible
}
```

## Function Scope (`var`)

`var` is function-scoped — accessible anywhere within the function.

```javascript
function test() {
    if (true) {
        var x = 10;
    }
    console.log(x);  // 10 (accessible outside the block)
}
```

## Block Scope (`let`, `const`)

`let` and `const` are block-scoped — only accessible within the block `{}`.

```javascript
function test() {
    if (true) {
        let y = 20;
        const z = 30;
    }
    console.log(y);  // ReferenceError: y is not defined
    console.log(z);  // ReferenceError: z is not defined
}
```

## Scope Chain

Inner functions can access variables from outer scopes:

```javascript
let global = "global";

function outer() {
    let outerVar = "outer";

    function inner() {
        let innerVar = "inner";
        console.log(global);   // accessible
        console.log(outerVar); // accessible
        console.log(innerVar); // accessible
    }

    inner();
    console.log(innerVar);  // ReferenceError
}
```

## Variable Shadowing

A variable in an inner scope can shadow one in an outer scope:

```javascript
let name = "Global";

function test() {
    let name = "Local";
    console.log(name);  // "Local"
}

test();
console.log(name);  // "Global"
```
