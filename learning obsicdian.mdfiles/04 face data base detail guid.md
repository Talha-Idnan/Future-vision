# FaceDatabase `load()` --- Detailed Guide

> \[!abstract\] Goal Understand how the database object finds its JSON
> file, reads its contents, and rebuilds `self.people` as a dictionary
> of NumPy arrays.

## 1. How the object is initialized

``` python
class FaceDatabase:
    def __init__(self, path=config.DB_FILE):
        self.path = path
        self.people = {}
        self.load()
```

When an object is created, Python calls `__init__()` to initialize it.

For example:

``` python
db1 = FaceDatabase()
db2 = FaceDatabase("test.json")
```

-   `db1` uses the default `config.DB_FILE`.
-   `db2` uses `"test.json"` because an argument was supplied.
-   `self.path` stores the chosen path on that object.
-   `self.people = {}` creates an empty dictionary for that object.
-   `self.load()` calls the `load()` method on that same object.

### Default parameter vs. supplied argument

``` python
def greet(name="Ali"):
    print(name)

greet()          # Ali
greet("Sara")    # Sara
```

A default is used only when the caller does not provide that argument.

### Three similar-looking names

  -----------------------------------------------------------------------
  Expression                          Meaning
  ----------------------------------- -----------------------------------
  `config.DB_FILE`                    Variable named `DB_FILE` inside the
                                      imported `config` module

  `path`                              Parameter/local name used while
                                      `__init__()` runs

  `self.path`                         Attribute stored on the current
                                      object
  -----------------------------------------------------------------------

`self.path = path` takes the value supplied to the parameter and stores
it on the object.

## 2. The complete `load()` method

``` python
def load(self):
    """Read the saved file from disk (if it exists)."""
    if not os.path.exists(self.path):
        return

    with open(self.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    self.people = {
        name: np.array(vec, dtype=np.float32)
        for name, vec in data.items()
    }
```

The method has three main stages:

``` mermaid
flowchart TD
    A[Start load()] --> B{Does self.path exist?}
    B -->|No| C[return: exit method]
    B -->|Yes| D[Open file in read mode]
    D --> E[json.load: parse JSON]
    E --> F[Convert lists to NumPy arrays]
    F --> G[Store rebuilt dictionary in self.people]
```

## 3. `os.path.exists(self.path)`

``` python
if not os.path.exists(self.path):
    return
```

-   `os` is Python's operating-system module.
-   `os.path` contains path-related helpers.
-   `exists(path)` checks whether something exists at that path.
-   It returns `True` if it exists and `False` if it does not.
-   `not` reverses that result.

So the condition means: **if the path does not exist, exit the method.**

This check does not create, open, or read the file. It only checks
existence.

### Why check first?

On a first run, the database file may not have been created yet. Trying
to open a missing file for reading would cause an error. The check
allows `load()` to exit safely.

## 4. `return`

``` python
if not os.path.exists(self.path):
    return
```

`return` immediately ends the current function or method.

Here, if the file is missing, Python leaves `load()` and does not
execute the file-opening or JSON-reading lines below it.

## 5. `with open(...) as f`

``` python
with open(self.path, "r", encoding="utf-8") as f:
    data = json.load(f)
```

Breakdown:

-   `open(...)` opens the file.
-   `self.path` tells Python which file to open.
-   `"r"` means read mode: read existing content, not write new content.
-   `encoding="utf-8"` tells Python how to decode the file's text.
-   `as f` gives the opened file object the variable name `f`.
-   `with` ensures the file is closed automatically when the indented
    block finishes, including if an error occurs.

`f` is not the file's text itself. It is the file object used to read
the file.

## 6. `json.load(f)`

``` python
data = json.load(f)
```

-   `json` is Python's JSON module.
-   `load()` reads JSON content from an opened file object.
-   It parses that JSON and returns normal Python data.
-   `data` stores the returned Python value.

If `database.json` contains:

``` json
{
  "Ali": [1, 2, 3],
  "Sara": [4, 5, 6]
}
```

After `json.load(f)`, `data` is a Python dictionary:

``` python
data = {
    "Ali": [1, 2, 3],
    "Sara": [4, 5, 6]
}
```

### `load()` vs. `loads()`

  Function             Reads from
  -------------------- --------------------------
  `json.load(f)`       An opened file object
  `json.loads(text)`   A string containing JSON

## 7. `data.items()`

A dictionary's `.items()` method lets you loop over its key-value pairs.

``` python
data = {"Ali": [1, 2, 3], "Sara": [4, 5, 6]}

for name, vec in data.items():
    print(name, vec)
```

Conceptually, the loop visits:

-   `name = "Ali"` and `vec = [1, 2, 3]`
-   `name = "Sara"` and `vec = [4, 5, 6]`

The key is the person's name. The value is that person's saved
fingerprint list.

## 8. Rebuilding `self.people`

``` python
self.people = {
    name: np.array(vec, dtype=np.float32)
    for name, vec in data.items()
}
```

This is a dictionary comprehension. It builds a new dictionary by
processing every key-value pair in `data`.

For each person:

1.  `name` receives the dictionary key, such as `"Ali"`.
2.  `vec` receives the saved list, such as `[1, 2, 3]`.
3.  `np.array(vec, dtype=np.float32)` converts that list into a NumPy
    array of 32-bit floating-point numbers.
4.  `name: converted_array` becomes an entry in the new dictionary.

The result is conceptually like:

``` python
self.people = {
    "Ali": np.array([1, 2, 3], dtype=np.float32),
    "Sara": np.array([4, 5, 6], dtype=np.float32)
}
```

### Why convert lists into NumPy arrays?

JSON stores arrays as ordinary lists. The face-recognition code uses
NumPy for numerical operations, so `load()` converts the saved lists
back into NumPy arrays.

In the actual project, each fingerprint contains 512 numbers. The short
example above uses only three numbers to make the idea easier to see.

## 9. The whole journey

``` text
JSON file on disk
      |
      | open(..., "r")
      v
File object f
      |
      | json.load(f)
      v
Python dictionary: data
      |
      | data.items()
      | convert each list with np.array(...)
      v
self.people dictionary of NumPy arrays
```

**Key distinction:** Loading changes the object's in-memory
`self.people`. It does not save anything back to the JSON file.

## 10. Check your understanding

Try answering before opening the answers.

1.  What value is used for `path` when `FaceDatabase()` is called
    without an argument?
2.  What does `os.path.exists(self.path)` check?
3.  What does `return` do if the file is missing?
4.  What does `"r"` mean in `open(self.path, "r")`?
5.  What is `f` in `with open(...) as f`?
6.  What does `json.load(f)` return?
7.  In `data.items()`, what do `name` and `vec` represent?
8.  Why does the project convert each list into
    `np.array(..., dtype=np.float32)`?

```{=html}
<details>
```
```{=html}
<summary>
```
Answers --- open after trying
```{=html}
</summary>
```
1.  The default value, `config.DB_FILE`.
2.  It checks whether the path exists.
3.  It immediately exits the `load()` method.
4.  Read mode.
5.  The file object returned by `open()`.
6.  The Python value represented by the JSON file; in this project, a
    dictionary.
7.  `name` is a person's key/name; `vec` is that person's saved
    fingerprint list.
8.  The project uses NumPy arrays for numerical processing, so the saved
    lists are converted back into arrays of 32-bit floats.

```{=html}
</details>
```

------------------------------------------------------------------------

**Next project step:** Study `save()` and compare how it converts NumPy
arrays into JSON-compatible lists before writing the database file.
