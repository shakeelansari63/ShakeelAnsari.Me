# Objects

Objects are collections of key-value pairs (properties).

## Creating Objects

```javascript
// Object literal
let user = {
    name: "Alice",
    age: 30,
    email: "alice@example.com"
};

// Using new Object()
let user = new Object();
user.name = "Alice";
```

## Accessing Properties

```javascript
let user = { name: "Alice", age: 30 };

user.name;           // "Alice" — dot notation
user["age"];         // 30     — bracket notation
let key = "name";
user[key];           // "Alice" — dynamic key
```

## Adding and Updating Properties

```javascript
let user = { name: "Alice" };
user.age = 30;               // add new property
user.name = "Bob";           // update existing
user["email"] = "bob@test.com";  // bracket notation
```

## Deleting Properties

```javascript
delete user.age;             // removes the property
```

## Method Shorthand

```javascript
let user = {
    name: "Alice",
    greet() {                // shorthand for greet: function() { ... }
        console.log("Hello, I'm " + this.name);
    }
};

user.greet();  // Hello, I'm Alice
```

## Computed Property Names

```javascript
let key = "dynamicKey";
let obj = {
    [key]: "dynamic value",
    ["prop_" + 1]: "value1"
};
```

## Checking if Property Exists

```javascript
let user = { name: "Alice", age: 30 };

"name" in user;      // true
"salary" in user;    // false
user.hasOwnProperty("name");  // true
user.salary !== undefined;    // false
```

## Object Keys, Values, Entries

```javascript
let user = { name: "Alice", age: 30, role: "admin" };

Object.keys(user);     // ["name", "age", "role"]
Object.values(user);   // ["Alice", 30, "admin"]
Object.entries(user);  // [["name", "Alice"], ["age", 30], ["role", "admin"]]
```

## Object Spread (ES2018)

```javascript
let user = { name: "Alice", age: 30 };
let details = { role: "admin", age: 31 };

let merged = { ...user, ...details };
// { name: "Alice", age: 31, role: "admin" }
// later properties override earlier ones
```
