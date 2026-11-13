# OpenCV — Drawing Bounding Boxes (`draw_box`)

## 1. Function Purpose

The `draw_box()` function draws:
- A rectangle around a detected face.
- A filled label bar above the rectangle containing a person's name.

## 2. Understanding Coordinates

```python
x1, y1, x2, y2 = map(int, box)
```

- `map(int, box)` converts all coordinates into integers.
- `x1, y1` = top-left corner.
- `x2, y2` = bottom-right corner.

**Important:**
- Width = `x2 - x1`
- Height = `y2 - y1`

## 3. Drawing the Face Rectangle

```python
cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
```

| Argument | Purpose |
|---|---|
| `img` | Image to draw on |
| `(x1, y1)` | Starting corner |
| `(x2, y2)` | Ending corner |
| `color` | Rectangle color (BGR) |
| `2` | Border thickness in pixels |

**Special rule:** Thickness `-1` fills the entire rectangle.

## 4. Measuring Text

```python
(text_w, text_h), _ = cv2.getTextSize(
    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
)
```

Measures how much space the text needs.

- `text` = Text to measure.
- `FONT_HERSHEY_SIMPLEX` = Font style.
- `0.6` = Font scale.
- `2` = Text thickness.
- `text_w` = Text width.
- `text_h` = Text height.
- `_` = Unused baseline value.

**Returned structure:**

`((width, height), baseline)`

## 5. Calculating Label Position

```python
top = max(0, y1 - text_h - 12)
```

Calculates the vertical position of the label bar.

- `y1` = Top of face rectangle.
- `text_h` = Text height.
- `12` = Extra vertical spacing.
- `max(0, ...)` prevents negative coordinates.

**Why?** Drawing outside the image boundaries can cause unwanted results.

## 6. Drawing the Filled Label Bar

```python
cv2.rectangle(
    img,
    (x1, top),
    (x1 + text_w + 10, y1),
    color,
    -1
)
```

- Starting point: `(x1, top)`
- Ending point: `(x1 + text_w + 10, y1)`
- `text_w` determines the label width.
- `10` adds horizontal padding.
- `-1` fills the entire label bar.

## Key Concepts to Remember

1. `map()` applies a function to every element.
2.