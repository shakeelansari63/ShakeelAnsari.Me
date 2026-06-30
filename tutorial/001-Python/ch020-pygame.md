# Pygame — Game Development

Pygame is a library for writing video games. Install with `pip install pygame`.

## Setup

```python
import pygame
import sys
from pygame import mixer

# Initialize Pygame
pygame.init()
```

## Create the Screen

```python
screen = pygame.display.set_mode((800, 600))
```

## Window Title and Icon

```python
pygame.display.set_caption("Game Name")

icon = pygame.image.load('icon.png')
pygame.display.set_icon(icon)
```

## The Event Loop

Pygame closes immediately after drawing. Use an infinite loop to keep it open.

```python
run_status = True
while run_status:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_status = False

    pygame.display.update()

pygame.quit()
sys.exit()
```

## Drawing Shapes

```python
# Rectangle (surface, color, (x, y, width, height), stroke)
pygame.draw.rect(screen, (255, 0, 0), (20, 20, 50, 50), 2)

# Line (surface, color, (start_x, start_y), (end_x, end_y), stroke)
pygame.draw.line(screen, (255, 0, 0), (20, 100), (30, 150), 5)
```

## Displaying Images

```python
screen.blit(pygame.image.load('player.png'), (x_axis, y_axis))
```

## Displaying Text

```python
font_attr = pygame.font.Font('font.ttf', 32)
text = 'Some Text'
text_attr = font_attr.render(text, True, (0, 0, 0))
screen.blit(text_attr, (x_pos, y_pos))
```

## Background Music

```python
mixer.music.load('background.wav')
mixer.music.play(-1)   # -1 loops indefinitely

# One-off sound effect
sound = mixer.Sound('sound.wav')
sound.play()
```

## Capturing Key Presses

```python
for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            # Move left
        if event.key == pygame.K_RIGHT:
            # Move right
```
