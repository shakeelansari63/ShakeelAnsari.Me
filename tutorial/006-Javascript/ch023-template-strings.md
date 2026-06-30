# Template Strings

Template strings (template literals) provide a powerful way to work with strings using backticks.

## Basic Syntax

```javascript
let name = "Alice";
let greeting = `Hello, ${name}!`;
console.log(greeting);  // Hello, Alice!
```

## Expressions Inside Templates

```javascript
let a = 10, b = 20;
console.log(`${a} + ${b} = ${a + b}`);  // 10 + 20 = 30
```

## Multi-Line Strings

```javascript
let html = `
<div>
    <h1>Title</h1>
    <p>This is a multi-line string</p>
</div>
`;

console.log(html);
```

## Function Calls in Templates

```javascript
function greet(name) {
    return "Hello, " + name + "!";
}

let user = "Alice";
console.log(`${greet(user)} Welcome to the site.`);
```

## Tagged Templates

A more advanced feature where a function processes the template:

```javascript
function highlight(strings, ...values) {
    let result = "";
    strings.forEach((str, i) => {
        result += str;
        if (i < values.length) {
            result += `<strong>${values[i]}</strong>`;
        }
    });
    return result;
}

let name = "Alice";
let role = "admin";
let output = highlight`User ${name} has role ${role}`;
console.log(output);  // User <strong>Alice</strong> has role <strong>admin</strong>
```
