# Future Vision — main.py (Imports to Initialization)

## 1. Purpose of main.py

`main.py` is the main application file. Its responsibility is to coordinate all components of Future Vision.

For every webcam frame, it:

1. Detects general objects using YOLO.
2. Detects faces using MTCNN.
3. Converts faces into numerical fingerprints using InceptionResnetV1.
4. Identifies people using FaceDatabase.
5. Draws bounding boxes and names using OpenCV.
6. Displays the processed frame.

**Important:** `main.py` coordinates the application. It does not implement every AI or database function itself.

---

## 2. Imports

Imports allow us to reuse functionality from existing Python modules and libraries.

| Import | Purpose |
|---|---|
| `time` | Measures processing time and FPS. |
| `Counter` | Finds the most common label. |
| `deque` | Stores recent labels with a limited history. |
| `cv2` | Webcam access, image processing and drawing. |
| `numpy` | Numerical calculations and arrays. |
| `torch` | Runs neural networks and handles tensors. |
| `MTCNN` | Detects faces. |
| `InceptionResnetV1` | Generates face embeddings. |
| `YOLO` | Detects general objects. |
| `FaceDatabase` | Reuses database functionality from face_db.py. |
| `SimpleNamespace` | Creates an object for storing configuration settings. |

### Why import FaceDatabase?

```python
from face_db import FaceDatabase
```

We import our own class instead of rewriting database functionality.

Benefits:
- Organization.
- Reusability.
- Maintainability.
- Separation of responsibilities.

Another Python file can reuse the same class:

```python
from face_db import FaceDatabase

db = FaceDatabase()
```

---

## 3. Configuration using SimpleNamespace

```python
config = SimpleNamespace(
    CAMERA_INDEX=0,
    FRAME_WIDTH=1280,
    FRAME_HEIGHT=720,
    FACE_MIN_CONFIDENCE=0.90,
    MIN_FACE_SIZE=60,
    RECOGNITION_MODEL="vggface2",
    MATCH_THRESHOLD=0.9,
    VOTE_WINDOW=7,
    SHOW_DISTANCE=True,
    MIN_SNAPSHOTS=3,
    YOLO_MODEL_PATH="yolo26n.pt",
    YOLO_CONFIDENCE=0.50,
    SHOW_PEOPLE_FROM_YOLO=False,
    DB_DIR="faces",
    DB_FILE="faces/database.json",
)
```

`SimpleNamespace` creates an object whose attributes can be accessed using dot notation.

Example:

```python
config.CAMERA_INDEX
config.FRAME_WIDTH
config.RECOGNITION_MODEL
```

### Configuration categories

| Category | Settings | Purpose |
|---|---|---|
| Camera | CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT | Camera selection and resolution. |
| Face detection | FACE_MIN_CONFIDENCE, MIN_FACE_SIZE | Face detection filtering. |
| Face recognition | RECOGNITION_MODEL, MATCH_THRESHOLD, VOTE_WINDOW, SHOW_DISTANCE, MIN_SNAPSHOTS | Recognition settings. |
| Object detection | YOLO_MODEL_PATH, YOLO_CONFIDENCE, SHOW_PEOPLE_FROM_YOLO | YOLO settings. |
| Storage | DB_DIR, DB_FILE | Database storage locations. |

**Why configuration?**

- Keeps settings organized.
- Makes tuning easier.
- Allows us to change behavior without modifying application logic.

---

## 4. Device Selection (CPU/GPU)

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")
```

`torch.cuda.is_available()` checks whether PyTorch can access a compatible CUDA GPU.

`torch.device()` creates a device object representing where calculations should run.

Logic:

- CUDA available → GPU.
- CUDA unavailable → CPU.

**Why?**

Automatic device selection allows the application to work on different computers without manually changing the code.

---

## 5. Model Initialization

Models are initialized once before the webcam loop because loading models is expensive.

### Model A — MTCNN

```python
face_detector = MTCNN(
    keep_all=True,
    device=device
)
```

- `MTCNN` is a class from facenet_pytorch.
- `face_detector` is its object (instance).
- `keep_all=True` returns all detected faces instead of only one.
- `device=device` specifies CPU or GPU.

**Responsibility:** WHERE is each face?

It returns face bounding boxes and confidence scores.

### Model B — InceptionResnetV1

```python
face_encoder = InceptionResnetV1(
    pretrained=config.RECOGNITION_MODEL
).eval().to(device)
```

**Responsibility:** WHO is the person?

It converts a face image into a 512-dimensional numerical embedding (fingerprint).

Understanding the arguments and methods:

| Code | Meaning |
|---|---|
| `pretrained=config.RECOGNITION_MODEL` | Selects pretrained weights, such as VGGFace2. |
| `.eval()` | Switches the model into evaluation mode. |
| `.to(device)` | Moves model parameters and buffers to the selected device. |

**Important distinctions:**

- Architecture = structure/design of the neural network.
- Weights = learned numerical parameters.
- Dataset = examples used during training.

`.eval()` does not itself stop gradient calculations. `torch.no_grad()` is used for that during inference.

### Model C — YOLO

```python
object_detector = YOLO(config.YOLO_MODEL_PATH)
```

- `YOLO` is a class from the Ultralytics library.
- `config.YOLO_MODEL_PATH` provides the model file path.
- `object_detector` stores the initialized model object.

**Responsibility:** Detect general objects such as phones, cups, chairs and cars.

The model path is configurable, so we can change the YOLO model without changing the initialization code.

---

## 6. Database Initialization

```python
db = FaceDatabase()
```

- `FaceDatabase` is our own class from face_db.py.
- `db` is an instance of that class.
- Its constructor initializes the database object and loads stored data according to the class implementation.

Now main.py can reuse its methods:

```python
db.identify(fingerprint)
db.add_person(name, pending)
db.delete_person(name)
db.names()
```

We do not need to rewrite those methods inside main.py.

---

## 7. Color Configuration (OpenCV BGR)

```python
COLOR_KNOWN = (120, 200, 0)
COLOR_UNKNOWN = (60, 60, 230)
COLOR_OBJECT = (255, 160, 0)
```

OpenCV uses BGR channel order:

```text
(BLUE, GREEN, RED)
```

These variables store colors for drawing rectangles and labels.

| Variable | BGR | Usage |
|---|---|---|
| COLOR_KNOWN | (120, 200, 0) | Recognized faces — green-ish. |
| COLOR_UNKNOWN | (60, 60, 230) | Unknown faces — red-ish. |
| COLOR_OBJECT | (255, 160, 0) | YOLO objects — blue-ish. |

### Why separate color variables?

Instead of repeating numerical color values throughout the program, we define them once.

Benefits:
- Readability.
- Maintainability.
- Easy color changes.
- Consistent appearance.

Example:

```python
cv2.rectangle(frame, point1, point2, COLOR_KNOWN, 2)
```

---

## 8. Complete Initialization Architecture

Before the webcam starts, main.py initializes:

| Variable | Component | Responsibility |
|---|---|---|
| `device` | PyTorch | Select computing device. |
| `face_detector` | MTCNN | Detect faces. |
| `face_encoder` | InceptionResnetV1 | Generate embeddings. |
| `object_detector` | YOLO | Detect objects. |
| `db` | FaceDatabase | Manage known people. |
| `COLOR_KNOWN` | Color setting | Recognized faces. |
| `COLOR_UNKNOWN` | Color setting | Unknown faces. |
| `COLOR_OBJECT` | Color setting | Detected objects. |

### Why initialize models before the webcam loop?

Model loading is expensive.

If models were initialized inside the webcam loop, they would repeatedly reload for every frame.

Instead, we initialize them once and reuse them for every frame.

---

## 9. Key Concepts to Remember

1. Imports allow code reuse.
2. SimpleNamespace organizes configuration settings.
3. Device selection chooses CPU or GPU.
4. MTCNN answers WHERE.
5. InceptionResnetV1 answers WHO.
6. YOLO detects general objects.
7. FaceDatabase manages saved people.
8. OpenCV uses BGR color ordering.
9. Models are initialized once, outside the webcam loop.
10. main.py coordinates all these components.

**Learning milestone:** Completed main.py imports, configuration, device selection, model initialization, database initialization and color configuration.

**Next topic:** Helper functions — starting with `make_fingerprint()`.
