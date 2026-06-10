# Tkinter — GUI Applications

Tkinter is Python's built-in GUI library. Create windows with widgets like Label, Button, Entry, and Listbox.

## Hello World

In Tkinter everything starts with a root widget. Then create and pack other widgets.

```python
import tkinter as tk

# Create the root window
root_widget = tk.Tk()

# Create a label and pack it
label = tk.Label(root_widget, text="Hello, World!")
label.pack()

# Start the event loop
root_widget.mainloop()
```

## Layout with Grid

Instead of `pack()`, use `grid()` to place widgets in rows and columns.

```python
import tkinter as tk

root_widget = tk.Tk()

label1 = tk.Label(root_widget, text="Hello, World!")
label2 = tk.Label(root_widget, text="I am Shakeel Ansari")

label1.grid(row=0, column=0)
label2.grid(row=1, column=1)

root_widget.mainloop()
```

## Buttons

Buttons can trigger functions with the `command` parameter.

```python
import tkinter as tk

def click_action(widget, row, column):
    tk.Label(widget, text="You clicked it!").grid(row=row, column=column)

root_widget = tk.Tk()

button1 = tk.Button(root_widget, text="Do Nothing", state=tk.DISABLED, padx=50, pady=50)
button1.grid(row=0, column=0)

button2 = tk.Button(root_widget, text="Do Something", padx=50, pady=50,
                    command=lambda: click_action(root_widget, 1, 0))
button2.grid(row=0, column=1)

root_widget.mainloop()
```

## Entry (Input Fields)

Entry widgets let the user type text. Use `.get()` to read the value.

```python
import tkinter as tk

def click_action():
    tk.Label(root_widget, text=inp.get()).grid(row=2, column=0, columnspan=2, sticky=tk.W)

root_widget = tk.Tk()

inp = tk.Entry(root_widget)
inp.grid(row=0, column=0, columnspan=2)

button = tk.Button(root_widget, text="Do Nothing", state=tk.DISABLED,
                   padx=50, pady=50, bg="Blue", fg="#ff0000")
button.grid(row=1, column=0)

button = tk.Button(root_widget, text="Do Something", padx=50, pady=50, command=click_action)
button.grid(row=1, column=1)

root_widget.mainloop()
```

## Listbox

Listbox shows a list of selectable items.

```python
from tkinter import *

top = Tk()

Lb1 = Listbox(top)
Lb1.insert(1, "Python")
Lb1.insert(2, "Perl")
Lb1.insert(3, "C")
Lb1.insert(4, "PHP")
Lb1.insert(5, "JSP")
Lb1.insert(6, "Ruby")

Lb1.pack()
top.mainloop()
```
