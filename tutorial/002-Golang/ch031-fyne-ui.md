# Fyne Desktop GUI

[Fyne](https://fyne.io/) is a cross-platform GUI toolkit for Go. It uses hardware-accelerated rendering and supports desktop, mobile, and embedded builds from a single codebase.

## Setup

```bash
go get fyne.io/fyne/v2@latest
go get fyne.io/fyne/v2/app@latest
go install fyne.io/fyne/v2/cmd/fyne@latest
```

The `fyne` CLI bundles applications for deployment. You can also run the demo to verify the toolkit works:

```bash
go run fyne.io/fyne/v2/cmd/fyne_demo@latest
```

## Minimal Window

```go
package main

import (
    "fyne.io/fyne/v2/app"
    "fyne.io/fyne/v2/widget"
)

func main() {
    a := app.New()
    w := a.NewWindow("Hello World")

    w.SetContent(widget.NewLabel("Hello World!"))
    w.ShowAndRun()
}
```

`app.New()` creates the application instance, which manages the event loop and lifecycle. `a.NewWindow("Hello World")` creates a window with the given title.

`w.SetContent()` sets the root widget of the window. Fyne widgets are composable — you can nest layouts (`vbox`, `hbox`, `grid`) and widgets (`label`, `button`, `entry`) to build complex UIs.

`w.ShowAndRun()` makes the window visible and starts the event loop. This call blocks until the window is closed.

## Adding Interactivity

Fyne widgets emit events through callbacks:

```go
w.SetContent(widget.NewButton("Click me", func() {
    log.Println("Button clicked")
}))
```

## Layouts

Use container layouts to arrange multiple widgets:

```go
import "fyne.io/fyne/v2/container"

content := container.NewVBox(
    widget.NewLabel("Hello!"),
    widget.NewButton("Quit", func() {
        a.Quit()
    }),
)
w.SetContent(content)
```

`container.NewVBox` stacks children vertically. `container.NewHBox`, `container.NewGridWithColumns`, and `container.NewBorder` are also available.

The Fyne event loop must run on the main thread. The `fyne` command handles packaging for each platform — producing `.app` bundles on macOS, `.exe` on Windows, and APKs on Android.
