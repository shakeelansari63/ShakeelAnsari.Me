# Arrays

Arrays store ordered collections of values. In JavaScript, arrays can hold mixed types.

## Creating Arrays

```javascript
let fruits = ["Apple", "Banana", "Orange"];
let mixed = [42, "hello", true, null, { name: "Alice" }];
let empty = [];
let numbers = new Array(1, 2, 3);   // rarely used
```

## Accessing Elements

```javascript
let fruits = ["Apple", "Banana", "Orange"];
fruits[0];           // "Apple"
fruits[1];           // "Banana"
fruits[fruits.length - 1];  // "Orange" (last element)
```

## Basic Operations

```javascript
let arr = [1, 2, 3];

arr.push(4);            // [1, 2, 3, 4]     — add to end
arr.pop();              // [1, 2, 3]         — remove from end
arr.unshift(0);         // [0, 1, 2, 3]      — add to start
arr.shift();            // [1, 2, 3]         — remove from start
arr.length;             // 3
```

## Iterating

```javascript
let numbers = [10, 20, 30];

// For loop
for (let i = 0; i < numbers.length; i++) {
    console.log(numbers[i]);
}

// For-of
for (let num of numbers) {
    console.log(num);
}

// ForEach
numbers.forEach((num, index) => {
    console.log(index + ": " + num);
});
```

## Spread Operator

```javascript
let a = [1, 2, 3];
let b = [...a, 4, 5];     // [1, 2, 3, 4, 5]
let c = [...a, ...b];     // [1, 2, 3, 1, 2, 3, 4, 5]
```
