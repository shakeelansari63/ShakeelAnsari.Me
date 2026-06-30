# Promises

A Promise represents a value that may be available now, later, or never.

## Promise States

- **Pending** — initial state, not fulfilled or rejected
- **Fulfilled** — operation completed successfully
- **Rejected** — operation failed

## Creating a Promise

```javascript
const fetchData = new Promise((resolve, reject) => {
    setTimeout(() => {
        const success = true;

        if (success) {
            resolve({ id: 1, name: "Alice" });
        } else {
            reject("Failed to fetch data");
        }
    }, 2000);
});
```

## Using a Promise

```javascript
fetchData
    .then((data) => {
        console.log("Data received:", data);
    })
    .catch((error) => {
        console.error("Error:", error);
    })
    .finally(() => {
        console.log("Promise settled (success or failure)");
    });
```

## Chaining Promises

```javascript
fetchUser(1)
    .then(user => fetchPosts(user.id))
    .then(posts => fetchComments(posts[0].id))
    .then(comments => console.log(comments))
    .catch(error => console.error(error));
```

## Promise All

Wait for multiple promises to complete:

```javascript
const p1 = fetchUser(1);
const p2 = fetchUser(2);
const p3 = fetchUser(3);

Promise.all([p1, p2, p3])
    .then(users => console.log("All users:", users))
    .catch(error => console.error("One failed:", error));
```

## Promise Race

Resolve or reject as soon as one promise settles:

```javascript
Promise.race([
    fetchFromServer(),
    fetchFromCache()
]).then(data => console.log("Fastest source:", data));
```

## Promise.allSettled

Wait for all promises, regardless of fulfillment or rejection:

```javascript
Promise.allSettled([p1, p2, p3])
    .then(results => {
        results.forEach(result => {
            if (result.status === "fulfilled") {
                console.log("Success:", result.value);
            } else {
                console.log("Failure:", result.reason);
            }
        });
    });
```
