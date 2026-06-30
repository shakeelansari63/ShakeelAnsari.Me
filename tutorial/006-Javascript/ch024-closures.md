# Closures

A closure is a function that remembers the variables from its outer scope even after the outer function has finished executing.

## Basic Closure

```javascript
function outer() {
    let message = "Hello from closure!";

    function inner() {
        console.log(message);  // remembers `message` from outer scope
    }

    return inner;
}

let myClosure = outer();
myClosure();  // Hello from closure!
```

The `inner` function "closes over" the `message` variable, keeping it alive.

## Practical: Counter

```javascript
function createCounter() {
    let count = 0;

    return {
        increment: function() { count++; },
        decrement: function() { count--; },
        getCount: function() { return count; }
    };
}

let counter = createCounter();
counter.increment();
counter.increment();
counter.increment();
console.log(counter.getCount());  // 3
// count is not directly accessible from outside
```

## Practical: Function Factory

```javascript
function createMultiplier(multiplier) {
    return function(number) {
        return number * multiplier;
    };
}

let double = createMultiplier(2);
let triple = createMultiplier(3);

console.log(double(5));   // 10
console.log(triple(5));   // 15
```

## Practical: Private Variables

```javascript
function createUser(name) {
    let _password = "";  // private

    return {
        getName: () => name,
        setPassword: (pwd) => { _password = pwd; },
        checkPassword: (pwd) => _password === pwd
    };
}

let user = createUser("Alice");
user.setPassword("secret123");
console.log(user.checkPassword("secret123"));  // true
console.log(user._password);  // undefined — private
```
