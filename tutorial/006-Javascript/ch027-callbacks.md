# Callbacks

A callback is a function passed as an argument to another function, to be executed later.

## Basic Callback

```javascript
function processUserInput(name, callback) {
    console.log("Processing user: " + name);
    callback(name);
}

processUserInput("Alice", function(name) {
    console.log("Hello, " + name + "!");
});
// Processing user: Alice
// Hello, Alice!
```

## Asynchronous Callback

```javascript
console.log("Start");

setTimeout(function() {
    console.log("This runs after 2 seconds");
}, 2000);

console.log("End");
// Output: Start, End, "This runs after 2 seconds"
```

## Callback with Array Methods

```javascript
let numbers = [1, 2, 3, 4, 5];

// map takes a callback
let doubled = numbers.map(function(n) {
    return n * 2;
});

// filter takes a callback
let evens = numbers.filter(function(n) {
    return n % 2 === 0;
});

// sort takes a callback
numbers.sort(function(a, b) {
    return b - a;  // descending
});
```

## Callback Hell

Nesting callbacks leads to deeply indented, hard-to-read code:

```javascript
getUser(1, function(user) {
    getPosts(user.id, function(posts) {
        getComments(posts[0].id, function(comments) {
            console.log(comments);
        });
    });
});
```

This is known as "callback hell." Promises and async/await solve this problem.
