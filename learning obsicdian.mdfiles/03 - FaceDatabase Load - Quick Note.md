# FaceDatabase --- Quick Reference

> \[!summary\] What this part does `__init__()` sets up a database
> object. `load()` checks for saved JSON data and, if found, loads it
> into `self.people`.

## 1. Default parameters

``` python
def __init__(self, path=config.DB_FILE):
```

-   `path` is a parameter.
-   `config.DB_FILE` is its default value.
-   `FaceDatabase()` uses the default.
-   `FaceDatabase("test.json")` overrides it.

## 2. `self` and object attributes

-   `self` means the current object.
-   `self.path = path` stores the path on that object.
-   `self.people = {}` creates its dictionary.
-   `config.DB_FILE`, `path`, and `self.path` are different: module
    variable, parameter, and object attribute.

## 3. The `load()` method

``` python
def load(self):
    if not os.path.exists(self.path):
        return

    with open(self.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    self.people = {
        name: np.array(vec, dtype=np.float32)
        for name, vec in data.items()
    }
```

-   `os.path.exists(...)` checks whether the file/path exists.
-   `return` exits `load()` if it does not exist.
-   `open(..., "r")` opens the file for reading.
-   `with` manages the file and closes it when the block ends.
-   `json.load(f)` reads JSON from the open file and converts it into
    Python data.
-   `data.items()` gives each dictionary key and value.
-   `np.array(..., dtype=np.float32)` converts each stored list into a
    NumPy array of 32-bit floats.
-   The dictionary comprehension rebuilds `self.people`.

## 4. JSON functions

-   `json.load(f)` reads JSON from a file object.
-   `json.loads(text)` reads JSON from a string.

## Detailed explanation

See \[\[02 - FaceDatabase load() - Detailed Guide\]\].
