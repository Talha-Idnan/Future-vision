# FaceDatabase --- Methods, Memory & JSON

> \[!abstract\] Main idea An object's attributes live in memory while
> the program runs. Saving and loading JSON are separate operations that
> move data between memory and a file.

## 1. Object data: RAM vs. disk

``` python
CD.people["Ali"] = [0.1, 0.2, 0.3]
```

This changes the dictionary held by the object while the program is
running.

-   **RAM (memory):** where the running program keeps its objects and
    their data.
-   **Disk (file):** storage that can remain after the program closes.
-   Changing `CD.people` does **not** automatically write to JSON.
-   A program must explicitly save data to a file and load it again
    later.

``` mermaid
flowchart LR
    A[JSON file on disk] -->|load| B[CD.people in memory]
    B -->|save| A
```

## 2. Methods and `self.load()`

A **method** is a function defined inside a class.

``` python
class FaceDatabase:
    def __init__(self, path):
        self.path = path
        self.people = {}
        self.load()

    def load(self):
        print("Loading from:", self.path)
```

-   `self.load()` calls `load()` for the current object.
-   Python passes that same object as `self`.
-   Therefore, `load()` can read `self.path` and update `self.people`.

Conceptually, `CD.load()` runs the method with `CD` as its `self`.

## 3. A small, runnable JSON example

This example uses only Python's standard library. Run it as a `.py`
file.

``` python
import json
from pathlib import Path


class FaceDatabase:
    def __init__(self, path):
        self.path = Path(path)
        self.people = {}
        self.load()

    def load(self):
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as file:
                self.people = json.load(file)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self.people, file, indent=4)

    def add_person(self, name, fingerprint):
        self.people[name] = fingerprint
        self.save()


# Create two objects with different file paths
CD1 = FaceDatabase("data/database1.json")
CD2 = FaceDatabase("data/database2.json")

CD1.add_person("Ali", [0.1, 0.2, 0.3])

print("CD1:", CD1.people)
print("CD2:", CD2.people)

# Create another object using CD1's file.
# Its __init__() calls load(), so it reads Ali from disk.
CD1_again = FaceDatabase("data/database1.json")
print("Reloaded:", CD1_again.people)
```

### What happens?

1.  `CD1` is initialized with its path and an empty dictionary.
2.  `self.load()` checks whether `database1.json` exists and loads it if
    it does.
3.  `CD2` does the same with a different path.
4.  `CD1.add_person(...)` changes `CD1.people` and calls `save()`.
5.  `CD1_again` loads the saved data from the same JSON file.

> \[!note\] JSON data types JSON supports objects, arrays, strings,
> numbers, booleans and `null`. Python dictionaries/lists/numbers map
> naturally to JSON. In the full face-recognition project, NumPy arrays
> need conversion to lists before `json.dump()`; they are not directly
> JSON-serializable.

## 4. Connecting this to your project

Your `face_db.py` follows the same general pattern:

-   `self.path` remembers which database file belongs to the object.
-   `self.people = {}` starts that object's in-memory dictionary.
-   `self.load()` reads saved people from the file, if it exists.
-   `save()` converts stored data into JSON-compatible values and writes
    the file.
-   `add_person()` updates the in-memory dictionary and saves it.

**Remember:** separate objects have separate instance dictionaries, but
two objects can load the same saved information if they use the same
file path.

## Quick recap

  Operation                     What it does
  ----------------------------- ---------------------------------------------
  `self.people = {}`            Creates an empty dictionary for this object
  `self.load()`                 Loads saved data into this object's memory
  `self.people[name] = value`   Changes the in-memory dictionary
  `self.save()`                 Writes the current data to disk
  `json.load(file)`             Reads JSON from a file
  `json.dump(data, file)`       Writes Python data as JSON

## Practice --- try before checking

1.  Does `CD.people["Ali"] = [...]` automatically save to disk?
2.  What does `self.load()` refer to when called inside `__init__()`?
3.  If `CD1` and `CD2` use different file paths, which file does each
    object's `self.path` identify?
4.  If two objects use the same JSON file, can both load the same saved
    people?
5.  Why does the full project convert NumPy arrays to lists before
    saving JSON?

```{=html}
<details>
```
```{=html}
<summary>
```
Answers
```{=html}
</summary>
```
1.  No. It changes the in-memory dictionary only, unless the code also
    calls `save()`.
2.  It calls the `load()` method on the object currently being
    initialized.
3.  Each object's own path, stored in its `self.path` attribute.
4.  Yes. Each object has its own dictionary, but both can independently
    load the same file contents.
5.  Standard JSON serialization does not directly support NumPy arrays;
    lists are JSON-compatible.

```{=html}
</details>
```

------------------------------------------------------------------------

**Previous note:** \[\[01 - Python OOP - Classes, Objects & self\]\]
