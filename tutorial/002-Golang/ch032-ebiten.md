# Ebitengine (Ebiten) 2D Game Engine

[Ebitengine](https://ebitengine.org/) (formerly Ebiten) is a simple 2D game engine for Go. It provides windowing, input handling, audio, and a rendering API — all in a single package.

## Setup

```bash
go get github.com/hajimehoshi/ebiten/v2
```

## The Game Interface

Ebitengine's main loop is driven by the `ebiten.Game` interface. To run a game you implement three methods:

```go
// ebiten.Game interface
type Game interface {
    Draw(screen *ebiten.Image)                         // render each frame
    Update() error                                     // update game state per tick
    Layout(outsideWidth, outsideHeight int) (int, int) // handle window resize
}
```

`Update` is called at a fixed rate (60 TPS by default). `Draw` is called after every update at the display's refresh rate. `Layout` tells the engine the game's logical screen size.

## Minimal Game Skeleton

```go
package main

import (
    "log"
    "github.com/hajimehoshi/ebiten/v2"
)

const (
    ScreenWidth  = 640
    ScreenHeight = 480
)

type SnakeGame struct{}

func (g *SnakeGame) Draw(screen *ebiten.Image) {}

func (g *SnakeGame) Update() error {
    return nil
}

func (g *SnakeGame) Layout(outsideWidth, outsideHeight int) (int, int) {
    return ScreenWidth, ScreenHeight
}

func main() {
    game := SnakeGame{}
    ebiten.SetWindowSize(ScreenWidth, ScreenHeight)
    ebiten.SetWindowTitle("Snake Game")
    if err := ebiten.RunGame(&game); err != nil {
        log.Fatal(err)
    }
}
```

`ebiten.SetWindowSize` and `ebiten.SetWindowTitle` configure the desktop window before starting the game.

`ebiten.RunGame` starts the main loop and blocks until the game exits. It takes a pointer to a struct that implements `Game`.

## Drawing

Inside `Draw`, you can render shapes, text, and images onto the `screen`:

```go
func (g *SnakeGame) Draw(screen *ebiten.Image) {
    screen.Fill(color.RGBA{0, 0, 0, 255}) // clear to black
}
```

## Game State

`Update` is where you handle input and advance game logic:

```go
func (g *SnakeGame) Update() error {
    if ebiten.IsKeyPressed(ebiten.KeyArrowLeft) {
        // move left
    }
    return nil
}
```

Returning a non-nil error from `Update` exits the game loop. This is how you'd implement a game-over condition.

Ebitengine runs on Windows, macOS, Linux, FreeBSD, Android, iOS, and WebAssembly — all from the same Go source.
