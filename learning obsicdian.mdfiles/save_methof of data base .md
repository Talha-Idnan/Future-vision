# Python Learning Notes — `save()` Method

## 1. Purpose
The `save()` method saves all known people's face fingerprints into `database.json` so the data remains available after the program closes.

## 2. Important Concepts

**1. Create the folder**
```python
os.makedirs(os.path.dirname(self.path), exist_ok=True)
```
- `os.path.dirname()` extracts the folder path (e.g. `"faces"`).
- `os.makedirs()` creates the folder if it doesn't exist.
- `exist_ok=True` prevents an error if the folder already exists.

**2. Convert NumPy arrays into lists**
```python
data = {name: vec.tolist() for name, vec in self.people.items()}
```
- `self.people.items()` provides each person's name and fingerprint.
- `vec.tolist()` converts NumPy arrays into normal Python lists.
- Dictionary comprehension creates a new dictionary called `data`.
- This conversion is needed because JSON cannot directly save NumPy arrays.

**3. Open the file**
```python
with open(self.path, "w", encoding="utf-8") as f:
```
- `"w"` means write mode.
- Creates the file if it doesn't exist.
- Overwrites existing file contents.
- `with` automatically closes the file.

**4. Save the data**
```python
json.dump(data, f, indent=2)
```
- `json.dump()` writes Python data into a JSON file.
- `data` is the dictionary being saved.
- `f` is the opened file.
- `indent=2` makes the JSON file readable and formatted.

## 3. Key Takeaway
The `save()` method ensures the folder exists, converts NumPy fingerprints into JSON-compatible lists, and saves the entire `self.people` dictionary into `database.json`.

### L2 Normalization
L2 normalization makes the **length (magnitude) of an entire vector equal to 1**.

- Square every value.
- Add all squared values.
- Take the square root (L2 norm).
- Divide every vector value by that norm.

**Formula:**

\[
\text{Normalized Vector}=\frac{\text{Vector}}{\text{L2 Norm}}
\]

In our `face_db.py`, it ensures every averaged face fingerprint has a consistent length of 1, making fingerprint comparisons more meaningful.