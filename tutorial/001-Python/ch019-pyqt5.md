# PyQt5 — Desktop GUI

PyQt5 is a powerful GUI library. Install with `pip install PyQt5`.

## Hello World

Create a `QApplication`, a `QMainWindow`, add a `QLabel`, and start the event loop.

```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt

class HelloWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("Hello World Program")

        label = QLabel("Hello World!!!")
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = HelloWindow()
    win.show()
    app.exec_()
```

## Key Concepts

- **QApplication**: Manages the application's event loop. Create it before any widgets.
- **QMainWindow**: The main window of the application. You can set a central widget.
- **QLabel**: Displays text or images. Use `setAlignment()` to control positioning.
- **exec_()**: Starts the event loop. The loop runs until the window is closed.
- **sys.argv**: Passed to QApplication to handle command-line arguments.
