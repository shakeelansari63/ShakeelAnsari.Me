# Break and Continue

## Break

`break` exits the loop immediately.

```javascript
for (let i = 0; i < 10; i++) {
    if (i === 5) {
        break;
    }
    console.log(i);
}
// Output: 0 1 2 3 4
```

## Continue

`continue` skips the current iteration and moves to the next.

```javascript
for (let i = 0; i < 10; i++) {
    if (i % 2 === 0) {
        continue;  // skip even numbers
    }
    console.log(i);
}
// Output: 1 3 5 7 9
```

## Break in While Loop

```javascript
let i = 0;
while (true) {
    if (i >= 5) {
        break;  // exit infinite loop
    }
    console.log(i);
    i++;
}
```

## Practical: Searching with Break

```javascript
let users = [
    { id: 1, name: "Alice" },
    { id: 2, name: "Bob" },
    { id: 3, name: "Charlie" }
];

let searchId = 2;
let foundUser = null;

for (let user of users) {
    if (user.id === searchId) {
        foundUser = user;
        break;  // stop searching once found
    }
}

console.log(foundUser);  // { id: 2, name: "Bob" }
```
