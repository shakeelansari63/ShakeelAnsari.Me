# Function Arguments

JavaScript functions can accept any number of arguments regardless of declared parameters.

## Accessing Arguments

```javascript
function showArgs() {
    console.log(arguments);  // array-like object of all passed arguments
    console.log(arguments[0]);
    console.log(arguments.length);
}

showArgs(1, 2, 3);
// Arguments(3) [1, 2, 3]
// 1
// 3
```

## Extra Arguments

```javascript
function add(a, b) {
    return a + b;
}

add(1, 2, 3, 4, 5);  // 3 — extra arguments are ignored
```

## Missing Arguments

```javascript
function greet(name, greeting) {
    return greeting + ", " + name + "!";
}

greet("Alice");  // "undefined, Alice!"
```

## Using Rest Parameters (modern)

```javascript
function multiply(multiplier, ...numbers) {
    return numbers.map(n => n * multiplier);
}

multiply(2, 1, 2, 3);  // [2, 4, 6]
```

## Arguments with Destructuring

```javascript
function createUser({ name, age, role = "user" }) {
    console.log(`${name} (${age}) - ${role}`);
}

createUser({ name: "Alice", age: 30 });
// Alice (30) - user

createUser({ name: "Bob", age: 25, role: "admin" });
// Bob (25) - admin
```
