# Array Properties and Methods

## Finding Elements

```javascript
let arr = [10, 20, 30, 40, 20];

arr.indexOf(20);            // 1
arr.lastIndexOf(20);        // 4
arr.includes(30);           // true
arr.find(x => x > 25);      // 30
arr.findIndex(x => x > 25); // 2
```

## Transforming

```javascript
let numbers = [1, 2, 3, 4, 5];

numbers.map(x => x * 2);          // [2, 4, 6, 8, 10]
numbers.filter(x => x % 2 === 0); // [2, 4]
numbers.reduce((sum, x) => sum + x, 0); // 15
```

## Sorting

```javascript
let arr = [3, 1, 4, 1, 5];
arr.sort();                       // [1, 1, 3, 4, 5] (lexicographic!)
arr.sort((a, b) => a - b);        // [1, 1, 3, 4, 5] (numeric ascending)
arr.sort((a, b) => b - a);        // [5, 4, 3, 1, 1] (descending)
arr.reverse();                    // reverse the array
```

## Slicing and Splicing

```javascript
let arr = [1, 2, 3, 4, 5];

arr.slice(1, 3);        // [2, 3] — extracts without modifying
arr.slice(-2);          // [4, 5] — last 2 elements

arr.splice(1, 2);       // [1, 4, 5]  — removes 2 elements from index 1
arr.splice(1, 0, 2, 3); // [1, 2, 3, 4, 5] — inserts at index 1
arr.splice(1, 1, 99);   // [1, 99, 3, 4, 5] — replaces 1 element
```

## Checking

```javascript
[1, 2, 3].every(x => x > 0);    // true — all pass
[1, 2, 3].some(x => x > 2);     // true — at least one passes
Array.isArray([1, 2, 3]);        // true
```

## Flattening

```javascript
[1, [2, 3], [4, [5, 6]]].flat();       // [1, 2, 3, 4, [5, 6]]
[1, [2, [3]]].flat(2);                  // [1, 2, 3]
[1, [2, [3, [4]]]].flat(Infinity);      // [1, 2, 3, 4]
```
