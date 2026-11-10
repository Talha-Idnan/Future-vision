# Main.py — Helper Functions: `make_fingerprint()`

## 1. Purpose of `make_fingerprint()`

`make_fingerprint()` converts one face image into a normalized face fingerprint containing **512 numerical values**.

It performs four main jobs:

1. Prepares the face image for the recognition model.
2. Converts the image into a PyTorch tensor.
3. Sends the image through the pretrained InceptionResnetV1 model.
4. Returns a normalized 512-dimensional embedding as a NumPy array.

**Important:** This function creates one fingerprint from one face image. It does not save the person in the database.

---

## 2. Complete Function

```python
def make_fingerprint(face_bgr):
    """Convert one face image into a normalized list of 512 numbers."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (160, 160))

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    tensor = (tensor - 127.5) / 128.0

    with torch.no_grad():
        embedding = face_encoder(tensor.unsqueeze(0).to(device))[0]

    embedding = embedding.cpu().numpy()
    return embedding / np.linalg.norm(embedding)
```

---

## 3. Step-by-Step Explanation

### Step 1 — Receive a face image

```python
def make_fingerprint(face_bgr):
```

- `def` creates a function.
- `make_fingerprint` is the function name.
- `face_bgr` is the parameter that receives a face image.

Example call:

```python
fingerprint = make_fingerprint(crop)
```

Here, `crop` is passed into the function and received by `face_bgr`.

### Step 2 — Convert BGR to RGB

```python
rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
```

OpenCV represents image colors in BGR order:

- Blue
- Green
- Red

Our face recognition model expects RGB:

- Red
- Green
- Blue

`cv2.cvtColor()` converts the color-channel order.

### Step 3 — Resize the image

```python
rgb = cv2.resize(rgb, (160, 160))
```

The InceptionResnetV1 model expects face images of 160 × 160 pixels.

This ensures every face image has a consistent size.

- Width = 160 pixels
- Height = 160 pixels

### Step 4 — Convert NumPy array into a PyTorch tensor

```python
tensor = torch.from_numpy(rgb)
```

OpenCV gives us an image represented as a NumPy array.

PyTorch models work with tensors.

`torch.from_numpy()` converts the NumPy array into a PyTorch tensor.

### Step 5 — Rearrange tensor dimensions

```python
tensor = tensor.permute(2, 0, 1)
```

OpenCV image shape:

```python
(160, 160, 3)
```

This means:

```text
(Height, Width, Channels)
```

PyTorch expects:

```text
(Channels, Height, Width)
```

After `permute(2, 0, 1)`:

```python
(3, 160, 160)
```

We have rearranged the dimensions, not changed the image resolution.

### Step 6 — Convert values into floating-point numbers

```python
tensor = tensor.float()
```

Converts the tensor's numerical values into floating-point values (`float32`).

This is suitable for the model's numerical calculations.

### Step 7 — Normalize pixel values

```python
tensor = (tensor - 127.5) / 128.0
```

Original image pixel values normally range from 0 to 255.

The model expects normalized pixel values approximately between -1 and +1.

The operation:

1. Subtracts 127.5 to shift the values around zero.
2. Divides by 128 to scale the values.

Example:

| Original value | Normalized value |
|---:|---:|
| 0 | -0.996 |
| 127.5 | 0 |
| 255 | 0.996 |

### Step 8 — Disable gradient calculations

```python
with torch.no_grad():
```

We are using pretrained model weights for inference, not training the model.

`torch.no_grad()` prevents unnecessary gradient calculations and saves memory.

**Remember the difference:**

- `.eval()` changes the model's evaluation behavior.
- `torch.no_grad()` disables gradient tracking.

### Step 9 — Add a batch dimension and generate the fingerprint

```python
embedding = face_encoder(tensor.unsqueeze(0).to(device))[0]
```

This line contains three important operations.

**A. `tensor.unsqueeze(0)`**

Before:

```python
(3, 160, 160)
```

After:

```python
(1, 3, 160, 160)
```

Why?

InceptionResnetV1 expects a four-dimensional input:

```text
(Batch Size, Channels, Height, Width)
```

The first dimension tells the model how many images are in the batch.

We add a batch dimension even though we are processing only one image.

`unsqueeze(0)` does not add another photograph. It adds a dimension.

**B. `.to(device)`**

Moves the tensor to the selected processing device.

The device was initialized earlier:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

- GPU if CUDA is available.
- Otherwise CPU.

**C. `face_encoder(...)`**

Sends the prepared face image through InceptionResnetV1.

The model produces an embedding containing 512 numerical values.

`[0]` selects the first result from the batch.

### Step 10 — Convert the embedding into a NumPy array

```python
embedding = embedding.cpu().numpy()
```

Two operations:

- `.cpu()` moves the tensor to CPU memory.
- `.numpy()` converts the CPU tensor into a NumPy array.

Why?

Our `face_db.py` uses NumPy operations such as `np.stack()`.

This allows the fingerprint to be processed by our database code.

The numerical values are preserved during this conversion.

### Step 11 — Normalize the fingerprint

```python
return embedding / np.linalg.norm(embedding)
```

`np.linalg.norm(embedding)` calculates the mathematical length (L2 norm) of the embedding vector.

Dividing the embedding by its length makes its magnitude equal to 1.

This is called L2 normalization.

Why?

It gives fingerprints a consistent vector length, making comparisons more suitable for face recognition.

The function returns the normalized NumPy array.

---

## 4. Understanding the Relationship with `add_person()`

These two functions perform different jobs.

### `make_fingerprint()` — main.py

Creates one fingerprint from one face image.

```python
fingerprint = make_fingerprint(crop)
```

### `add_person()` — face_db.py

Receives multiple fingerprints, averages them, normalizes the average, and saves one representative fingerprint for the person.

```python
db.add_person(name, pending)
```

### Example

Suppose we capture Ali's face five times.

```text
Face snapshot 1 → make_fingerprint() → Fingerprint 1
Face snapshot 2 → make_fingerprint() → Fingerprint 2
Face snapshot 3 → make_fingerprint() → Fingerprint 3
Face snapshot 4 → make_fingerprint() → Fingerprint 4
Face snapshot 5 → make_fingerprint() → Fingerprint 5
```

All five fingerprints are collected in `pending`.

Then:

```python
db.add_person("Ali", pending)
```

`add_person()` combines the five fingerprints into one average fingerprint and saves it.

**Remember:**

- 5 face snapshots.
- 5 individual fingerprints.
- 1 final representative fingerprint saved for Ali.

---

## 5. Complete Workflow

```text
One face image
      |
      v
Convert BGR to RGB
      |
      v
Resize to 160 × 160
      |
      v
Convert NumPy array to