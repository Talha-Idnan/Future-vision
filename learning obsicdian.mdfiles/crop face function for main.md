# `crop_face()` — Cropping a Face from an Image

## 1. Purpose

The `crop_face()` function extracts a detected face from the original camera image.

It also adds a small margin around the face so that important facial areas, such as hair and chin, are not accidentally cut off.

**Input:**
- `frame`: Original camera image.
- `box`: Coordinates of the detected face.
- `margin`: Extra space around the face (default: 15%).

**Output:**
- A cropped image containing the face and some surrounding space.

---

## 2. Complete Code

```python
def crop_face(frame, box, margin=0.15):
    """Cut the face out of the frame, with a small margin so hair and chin are included."""

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    pad_x = int((x2 - x1) * margin)
    pad_y = int((y2 - y1) * margin)

    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)

    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    return frame[y1:y2, x1:x2]
```

---

## 3. Understanding the Coordinates

The face detection model provides four coordinates:

```python
box = (x1, y1, x2, y2)
```

| Coordinate | Meaning |
|---|---|
| `x1` | Left boundary |
| `y1` | Top boundary |
| `x2` | Right boundary |
| `y2` | Bottom boundary |

Think of these coordinates as the four boundaries of a rectangle surrounding the detected face.

## 4. Line-by-Line Explanation

### Step 1: Get image dimensions

```python
h, w = frame.shape[:2]
```

An image's shape generally looks like:

```python
(height, width, channels)
```

Example:

```python
frame.shape = (480, 640, 3)
```

Therefore:

```python
h = 480
w = 640
```

**Why?**

We need the original image boundaries to prevent our crop from going outside the image.

### Step 2: Convert coordinates into integers

```python
x1, y1, x2, y2 = map(int, box)
```

The `map(int, box)` converts every coordinate into an integer.

Example:

```python
box = (100.7, 50.2, 300.9, 250.6)
```

Becomes:

```python
x1 = 100
y1 = 50
x2 = 300
y2 = 250
```

**Why?**

Image slicing uses integer pixel positions.

### Step 3: Calculate padding

```python
pad_x = int((x2 - x1) * margin)
pad_y = int((y2 - y1) * margin)
```

First, calculate the face dimensions:

```python
face_width = x2 - x1
face_height = y2 - y1
```

Then multiply them by the margin.

Example:

```python
x1 = 100
y1 = 50
x2 = 300
y2 = 250

margin = 0.15
```

Calculations:

```python
pad_x = int(200 * 0.15)  # 30
pad_y = int(200 * 0.15)  # 30
```

**Why?**

We want to expand the face rectangle by 15% of its width and height.

- `pad_x`: Extra space on the left and right.
- `pad_y`: Extra space on the top and bottom.

### Step 4: Expand the top-left boundaries

```python
x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
```

We subtract padding to move the left and top boundaries outward.

Example:

```python
x1 = 100 - 30  # 70
y1 = 50 - 30   # 20
```

**Why use `max(0, ...)`?**

Imagine:

```python
x1 = 10
pad_x = 30
```

Subtracting gives:

```python
10 - 30 = -20
```

But negative image coordinates are outside the image.

Therefore:

```python
max(0, -20)
```

Returns:

```python
0
```

`max(0, ...)` prevents negative coordinates.

### Step 5: Expand the bottom-right boundaries

```python
x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
```

We add padding to expand the right and bottom boundaries.

Example:

```python
x2 = 300 + 30  # 330
y2 = 250 + 30  # 280
```

**Why use `min()`?**

Imagine:

```python
w = 640
x2 = 630
pad_x = 30
```

Adding padding gives:

```python
630 + 30 = 660
```

But the image width is only 640 pixels.

Therefore:

```python
min(640, 660)
```

Returns:

```python
640
```

`min()` prevents coordinates from exceeding the image boundaries.

### Step 6: Actually crop the face

```python
return frame[y1:y2, x1:x2]
```

This is the most important line.

NumPy image slicing follows this structure:

```python
image[rows, columns]
```

Remember:

- Rows = Y coordinates.
- Columns = X coordinates.

Using our example:

```python
x1 = 70
y1 = 20
x2 = 330
y2 = 280
```

The slicing becomes:

```python
return frame[20:280, 70:330]
```

This extracts the rectangular portion of the original image.

**Important:** The ending index is excluded in Python slicing.

---

## 5. Complete Numerical Example

Suppose:

```python
frame.shape = (480, 640, 3)

box = (100, 50, 300, 250)

margin = 0.15
```

| Operation | Result |
|---|---:|
| Image height | 480 |
| Image width | 640 |
| Face width | 200 |
| Face height | 200 |
| Horizontal padding | 30 |
| Vertical padding | 30 |
| New `x1` | 70 |
| New `y1` | 20 |
| New `x2` | 330 |
| New `y2` | 280 |

Final slicing:

```python
frame[20:280, 70:330]
```

Resulting cropped image dimensions:

```python
(260, 260, 3)
```

---

## 6. Important Python Concepts Used

| Concept | Purpose |
|---|---|
| `frame.shape` | Gets image dimensions |
| `[:2]` | Gets the first two shape values |
| `map(int, box)` | Converts coordinates into integers |
| `int()` | Converts a number into an integer |
| `max()` | Prevents negative coordinates |
| `min()` | Prevents coordinates exceeding image boundaries |
| NumPy slicing | Extracts a portion of an image |
| `return` | Sends the cropped image back to the caller |

---

## 7. Visual Workflow

```text
Original Camera Image
        |
        v
Receive Face Coordinates
        |
        v
Calculate Face Width and Height
        |
        v
Calculate Extra Padding
        |
        v
Expand Face Rectangle
        |
        v
Check Image Boundaries
   max() and min()
        |
        v
Crop Using NumPy Slicing
        |
        v
Return Cropped Face
```

---

## 8. Important Things to Remember

1. `x2 - x1` calculates face width.
2. `y2 - y1` calculates face height.
3. `pad_x` controls horizontal extra space.
4. `pad_y` controls vertical extra space.
5. `max()` protects the left and top boundaries.
6. `min()` protects the right and bottom boundaries.
7. NumPy slicing uses Y first and X second.

## 9. Connection to Future Vision

In Future Vision, this function prepares the detected face for further processing.

The general workflow is:

```text
Camera Frame
     |
     v
Face Detection
     |
     v
Bounding Box Coordinates
     |
     v
crop_face()
     |
     v
Cropped Face Image
     |
     v
Further Face Processing
```

**Main takeaway:** `crop_face()` converts face coordinates into an actual cropped image while adding extra space and respecting the original image boundaries.

---

## 10. Self-Assessment

Before considering this function fully understood, make sure you can:

- [ ] Explain what `frame`, `box`, and `margin` represent.
- [ ] Explain why we use `frame.shape[:2]`.
- [ ] Calculate `pad_x` and `pad_y` manually.
- [ ] Explain why we subtract padding from `x1` and `y1`.
- [ ] Explain why we add padding to `x2` and `y2`.
- [ ] Explain the difference between `max()` and `min()`.
- [ ] Explain why NumPy slicing uses `frame[y1:y2, x1:x2]`.
- [ ] Write the complete function without looking at the original code.

**Milestone:** Understand how a bounding box becomes an actual cropped face image.