# Conditional Statements

## If-Else

```javascript
let score = 85;

if (score >= 90) {
    console.log("Grade: A");
} else if (score >= 80) {
    console.log("Grade: B");
} else if (score >= 70) {
    console.log("Grade: C");
} else {
    console.log("Grade: F");
}
```

## Ternary Operator

```javascript
let age = 20;
let status = age >= 18 ? "Adult" : "Minor";
```

## Switch Statement

```javascript
let day = 3;
let dayName;

switch (day) {
    case 1:
        dayName = "Monday";
        break;
    case 2:
        dayName = "Tuesday";
        break;
    case 3:
        dayName = "Wednesday";
        break;
    case 4:
        dayName = "Thursday";
        break;
    case 5:
        dayName = "Friday";
        break;
    default:
        dayName = "Weekend";
}
```

## Truthy and Falsy in Conditions

```javascript
if ("hello") { }       // truthy
if (0) { }             // falsy
if (null) { }          // falsy
if (undefined) { }     // falsy
if ([]) { }            // truthy (empty array)
if ({}) { }            // truthy (empty object)
```

## Logical Operators

```javascript
true && false;    // false
true || false;    // true
!true;            // false

// Short-circuit evaluation
let name = user.name || "Guest";   // default value
let isAdmin = user.role && (user.role === "admin");
```
