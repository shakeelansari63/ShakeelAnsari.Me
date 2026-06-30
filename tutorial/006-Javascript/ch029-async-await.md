# Async/Await

Async/await is syntactic sugar over Promises, making asynchronous code read like synchronous code.

## Async Function

```javascript
async function fetchData() {
    return "data";
}

fetchData().then(data => console.log(data));  // "data"
```

An `async` function always returns a Promise.

## Await

`await` pauses execution until the Promise settles:

```javascript
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function getUsers() {
    console.log("Fetching users...");
    await delay(2000);           // wait 2 seconds
    console.log("Users fetched!");
    return [{ id: 1, name: "Alice" }];
}

getUsers().then(users => console.log(users));
```

## Error Handling

```javascript
async function fetchUser(id) {
    try {
        const response = await fetch(`/api/users/${id}`);
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }
        const user = await response.json();
        return user;
    } catch (error) {
        console.error("Failed to fetch user:", error);
        return null;
    }
}
```

## Sequential vs Parallel

```javascript
// Sequential (one after another) — slower
async function getSequential() {
    const user = await fetchUser(1);
    const posts = await fetchPosts(user.id);
    const comments = await fetchComments(posts[0].id);
    return comments;
}

// Parallel (all at once) — faster
async function getParallel() {
    const [user1, user2, user3] = await Promise.all([
        fetchUser(1),
        fetchUser(2),
        fetchUser(3)
    ]);
    return [user1, user2, user3];
}
```

## Comparison

```javascript
// Callback hell
getUser(1, function(user) {
    getPosts(user.id, function(posts) {
        getComments(posts[0].id, function(comments) {
            console.log(comments);
        });
    });
});

// Promises
getUser(1)
    .then(user => getPosts(user.id))
    .then(posts => getComments(posts[0].id))
    .then(comments => console.log(comments))
    .catch(error => console.error(error));

// Async/await (cleanest)
async function displayComments() {
    try {
        const user = await getUser(1);
        const posts = await getPosts(user.id);
        const comments = await getComments(posts[0].id);
        console.log(comments);
    } catch (error) {
        console.error(error);
    }
}
```
