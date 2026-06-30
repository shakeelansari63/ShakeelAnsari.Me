# DOM Event Handlers

Events allow JavaScript to respond to user interactions.

## Adding Event Listeners

```javascript
let button = document.querySelector('button');

button.addEventListener('click', function(event) {
    console.log('Button was clicked!');
    console.log(event);  // MouseEvent object
});
```

## Common Events

| Event | Description |
|-------|-------------|
| `click` | Element clicked |
| `dblclick` | Element double-clicked |
| `mouseenter` | Mouse enters element |
| `mouseleave` | Mouse leaves element |
| `keydown` | Key pressed down |
| `keyup` | Key released |
| `submit` | Form submitted |
| `load` | Page/element finished loading |
| `scroll` | Element scrolled |
| `change` | Input value changed |
| `input` | Input value changed (real-time) |

## The Event Object

```javascript
document.querySelector('a').addEventListener('click', function(event) {
    event.preventDefault();   // prevent default behavior (e.g., navigation)
    console.log('Link clicked but navigation prevented');
});

document.querySelector('div').addEventListener('click', function(event) {
    event.stopPropagation();  // prevent event from bubbling up
    console.log('Event propagation stopped');
});
```

## Event Delegation

Attach a single listener to a parent instead of many listeners to children:

```javascript
document.querySelector('ul').addEventListener('click', function(event) {
    if (event.target.tagName === 'LI') {
        console.log('You clicked item:', event.target.textContent);
    }
});
```

## Removing Event Listeners

```javascript
function handler() {
    console.log('Clicked');
}

button.addEventListener('click', handler);
button.removeEventListener('click', handler);  // must be the same function reference
```

## Keyboard Events

```javascript
document.addEventListener('keydown', function(event) {
    console.log('Key pressed:', event.key);
    console.log('Key code:', event.code);

    if (event.key === 'Enter') {
        console.log('Enter was pressed');
    }

    if (event.ctrlKey && event.key === 's') {
        event.preventDefault();
        console.log('Ctrl+S was pressed');
    }
});
```
