# Loop Examples

## Sum of Array

```javascript
let numbers = [10, 20, 30, 40, 50];
let sum = 0;

for (let i = 0; i < numbers.length; i++) {
    sum += numbers[i];
}

console.log("Sum:", sum);  // 150
```

## Finding Maximum

```javascript
let scores = [85, 92, 78, 95, 88];
let max = scores[0];

for (let i = 1; i < scores.length; i++) {
    if (scores[i] > max) {
        max = scores[i];
    }
}

console.log("Max:", max);  // 95
```

## Filtering

```javascript
let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let evens = [];

for (let num of numbers) {
    if (num % 2 === 0) {
        evens.push(num);
    }
}

console.log(evens);  // [2, 4, 6, 8, 10]
```

## Frequency Counter

```javascript
let text = "hello world hello javascript hello world";
let words = text.split(" ");
let frequency = {};

for (let word of words) {
    if (frequency[word]) {
        frequency[word]++;
    } else {
        frequency[word] = 1;
    }
}

console.log(frequency);
// { hello: 3, world: 2, javascript: 1 }
```

## Nested Loop: Multiplication Table

```javascript
for (let i = 1; i <= 5; i++) {
    let row = "";
    for (let j = 1; j <= 5; j++) {
        row += (i * j).toString().padStart(4, " ");
    }
    console.log(row);
}
```
