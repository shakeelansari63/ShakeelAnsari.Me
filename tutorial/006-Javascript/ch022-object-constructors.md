# Object Constructors

Object constructors are like blueprints for creating multiple similar objects.

## Constructor Function

```javascript
function User(name, age) {
    this.name = name;
    this.age = age;
    this.greet = function() {
        console.log("Hello, I'm " + this.name);
    };
}

let alice = new User("Alice", 30);
let bob = new User("Bob", 25);

alice.greet();  // Hello, I'm Alice
bob.greet();    // Hello, I'm Bob
```

## The `new` Keyword

When you call `new User(...)`, JavaScript:

1. Creates a new empty object
2. Sets `this` to the new object
3. Links the object to the constructor's prototype
4. Returns the object (unless constructor returns another object)

## Constructor with Prototype Methods

For better performance, define methods on the prototype instead of inside the constructor:

```javascript
function User(name, age) {
    this.name = name;
    this.age = age;
}

User.prototype.greet = function() {
    console.log("Hello, I'm " + this.name);
};

User.prototype.isAdult = function() {
    return this.age >= 18;
};

let alice = new User("Alice", 30);
alice.greet();   // Hello, I'm Alice
```

## Class Syntax (ES6)

ES6 classes provide a cleaner syntax for constructors and prototypes:

```javascript
class User {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }

    greet() {
        console.log("Hello, I'm " + this.name);
    }

    isAdult() {
        return this.age >= 18;
    }
}

let alice = new User("Alice", 30);
alice.greet();  // Hello, I'm Alice
```

## Static Methods

```javascript
class User {
    constructor(name) {
        this.name = name;
    }

    static createGuest() {
        return new User("Guest");
    }
}

let guest = User.createGuest();  // called on the class, not an instance
```
