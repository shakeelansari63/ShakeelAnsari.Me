# DOM Manipulation

The Document Object Model (DOM) represents the HTML page as a tree of nodes that JavaScript can manipulate.

## Selecting Elements

```javascript
// By CSS selector (returns first match)
let node = document.querySelector('.masthead li');
console.log(node);

// By CSS selector (returns all matches in an array-like NodeList)
let nodes = document.querySelectorAll('.masthead li');
console.log(nodes);

// By ID
document.getElementById('content');

// By class name
document.getElementsByClassName('menu');

// By tag name
document.getElementsByTagName('li');
```

## Accessing Content

```javascript
let node = document.querySelector('.masthead li a');

node.innerHTML;               // text content including HTML tags
node.outerHTML;                // HTML including the element itself
node.textContent;              // just the text
```

## Attributes

```javascript
node.hasAttribute('href');     // true/false
node.getAttribute('href');     // get value
node.setAttribute('href', '#home');  // set value
node.removeAttribute('href');  // remove
```

## Class Manipulation

```javascript
let nav = document.querySelector('.masthead');
nav.classList;                 // DOMTokenList of classes
nav.classList.add('custom-class');
nav.classList.remove('clear');
nav.classList.toggle('active');
nav.classList.contains('menu'); // true/false
```

## Creating Elements

```javascript
let list = document.querySelector('.masthead ul');
let newItem = document.createElement('li');
let text = document.createTextNode('New Item');
newItem.appendChild(text);
list.appendChild(newItem);
```

## Modifying Styles

```javascript
let el = document.querySelector('.masthead');
el.style.backgroundColor = '#d53a9d';
el.style.color = 'white';
el.style.display = 'flex';
```

## Event Basics

```javascript
document.querySelector('button').addEventListener('click', function() {
    console.log('Button clicked!');
});
```
