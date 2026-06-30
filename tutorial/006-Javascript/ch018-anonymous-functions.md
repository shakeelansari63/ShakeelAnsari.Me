# Anonymous Functions

Anonymous functions have no name. They are often used as arguments to other functions or assigned to variables.

## Assigned to a Variable

```javascript
const greet = function(name) {
    return "Hello, " + name + "!";
};
```

## Used as Callback

```javascript
setTimeout(function() {
    console.log("This runs after 1 second");
}, 1000);

[1, 2, 3].forEach(function(num) {
    console.log(num * 2);
});
```

## Arrow Function Anonymous

```javascript
const doubled = [1, 2, 3].map(n => n * 2);
```

## Event Handlers

```javascript
document.querySelector("button").addEventListener("click", function() {
    console.log("Button clicked!");
});
```

Anonymous functions are useful when you need a one-time function that doesn't need to be reused elsewhere.
