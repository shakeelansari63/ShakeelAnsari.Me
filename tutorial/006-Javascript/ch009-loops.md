# Loops

## For Loop

```javascript
for (let i = 0; i < 5; i++) {
    console.log("Iteration: " + i);
}
```

## For-Of Loop (Arrays, Strings, etc.)

```javascript
let fruits = ["Apple", "Banana", "Orange"];
for (let fruit of fruits) {
    console.log(fruit);
}

let name = "JavaScript";
for (let char of name) {
    console.log(char);
}
```

## For-In Loop (Object Properties)

```javascript
let user = { name: "Alice", age: 30, role: "admin" };
for (let key in user) {
    console.log(key + ": " + user[key]);
}
```

## While Loop

```javascript
let i = 0;
while (i < 5) {
    console.log(i);
    i++;
}
```

## Do-While Loop

```javascript
let i = 0;
do {
    console.log(i);
    i++;
} while (i < 5);
```

## Looping Arrays with forEach

```javascript
[10, 20, 30].forEach(function(value, index) {
    console.log(index + ": " + value);
});

// Arrow syntax
[10, 20, 30].forEach((value, index) => console.log(index + ": " + value));
```

## Practical Loop Example

```javascript
// Print multiplication table
for (let i = 1; i <= 10; i++) {
    let row = "";
    for (let j = 1; j <= 10; j++) {
        row += (i * j).toString().padStart(4, " ");
    }
    console.log(row);
}
```
