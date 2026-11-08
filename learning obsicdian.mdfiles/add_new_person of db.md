# 🧠 Python Notes — `add_person()` Method

## 1. 🎯 Purpose of `add_person()`

The `add_person()` method takes multiple face fingerprints (snapshots), combines them into **one average fingerprint**, normalizes it, stores it in the dictionary, and saves everything to a JSON file.

```python
def add_person(self, name, fingerprints):
    stacked = np.stack(fingerprints)
    average = stacked.mean(axis=0)
    average /= np.linalg.norm(average)
    self.people[name] = average
    self.save()
```

---

## 2. 📥 What goes inside the function?

- `self` → The current object.
- `name` → The person's name, e.g. `"Talha"`.
- `fingerprints` → A list of NumPy arrays, each containing one face fingerprint.

For easy understanding, imagine we have **3 snapshots**, each containing **4 numbers**.

```python
fingerprints = [
    [1, 2, 3, 4],       # Snapshot 1
    [2, 4, 6, 8],       # Snapshot 2
    [3, 6, 9, 12]       # Snapshot 3
]
```

*In the real project, each fingerprint contains 512 numbers, not 4.*

---

## 3. 🧱 Step One — `np.stack(fingerprints)`

```python
stacked = np.stack(fingerprints)
```

`np.stack()` combines separate arrays into one larger 2D array.

**Visual representation:**

```text
                   FEATURES (COLUMNS)
                   1    2    3    4
                 ┌──────────────────
SNAPSHOT 1  →    │ 1    2    3    4
SNAPSHOT 2  →    │ 2    4    6    8
SNAPSHOT 3  →    │ 3    6    9   12
                 └──────────────────
```

- Rows = different snapshots.
- Columns = feature positions inside each fingerprint.
- `axis=0` = row direction.
- `axis=1` = column direction.

**Shape:**

```python
stacked.shape

(3, 4)
```

This means:
- 3 rows (snapshots).
- 4 columns (features).

---

## 4. 📊 Step Two — `mean(axis=0)`

```python
average = stacked.mean(axis=0)
```

This calculates the average **down each column**.

```text
Column 1: (1 + 2 + 3) / 3 = 2

Column 2: (2 + 4 + 6) / 3 = 4

Column 3: (3 + 6 + 9) / 3 = 6

Column 4: (4 + 8 + 12) / 3 = 8
```

**Result:**

```python
average = [2, 4, 6, 8]
```

**Shape changes:**

```text
Before: (3, 4)  →  3 snapshots × 4 features

After:  (4,)    →  1 average fingerprint
```

💡 **Remember:** `axis=0` averages down the rows, producing one average value for each column.

---

## 5. 📏 Step Three — `np.linalg.norm(average)`

```python
np.linalg.norm(average)
```

This calculates the **length (magnitude)** of the vector.

For our example:

```text
average = [2, 4, 6, 8]

length = √(2² + 4² + 6² + 8²)

       = √120

       ≈ 10.954
```

The result is one number, not another array of fingerprints.

---

## 6. ⚖️ Step Four — Normalize the Vector

```python
average /= np.linalg.norm(average)
```

This divides every value by the vector's length.

```text
BEFORE NORMALIZATION:

[2, 4, 6, 8]

       ÷ 10.954

AFTER NORMALIZATION:

[0.1826, 0.3651, 0.5477, 0.7303]
```

The vector now has a length of approximately **1**.

### Important difference

| Code | Purpose |
|---|---|
| `.mean(axis=0)` | Calculates the average |
| `np.linalg.norm()` | Calculates vector length |
| `/=` | Divides values by the length |
| Normalization | Makes vector length equal to 1 |

**Remember:** Normalization changes the numerical values, not the shape.

```text
Before: (4,)
After:  (4,)
```

---

## 7. 📦 Step Five — Store the Fingerprint

```python
self.people[name] = average
```

This stores the final fingerprint in the `people` dictionary.

Imagine:

```python
self.people["Talha"] = [0.1826, 0.3651, 0.5477, 0.7303]
```

### What if the person already exists?

Python dictionary assignment replaces the old value for that key.

```text
BEFORE:

people = {
    "Talha": old_fingerprint,
    "Ali": ali_fingerprint
}

AFTER registering Talha again:

people = {
    "Talha": new_fingerprint,
    "Ali": ali_fingerprint
}
```

Only Talha's old fingerprint is replaced. Ali's data remains.

**Important:** This method averages only the fingerprints passed into the current call. It does not automatically combine new snapshots with previously saved snapshots.

---

## 8. 💾 Step Six — Save the Database

```python
self.save()
```

This calls another method named `save()` on the same object.

Its job is to write the current `self.people` dictionary into the JSON file.

In the real project, JSON stores one final fingerprint containing **512 numbers per person**, not every original snapshot.

---

## 9. 🔄 Complete Visual Flow

```text
       📸 MULTIPLE FACE SNAPSHOTS
                  |
                  ▼
       np.stack(fingerprints)
                  |
                  ▼
          2D ARRAY (3, 4)
       3 rows × 4 columns
                  |
                  ▼
          mean(axis=0)
                  |
                  ▼
       AVERAGE VECTOR (4,)
                  |
                  ▼
       np.linalg.norm(average)
                  |
                  ▼
          VECTOR LENGTH
             10.954
                  |
                  ▼
       average /= vector length
                  |
                  ▼
       NORMALIZED VECTOR (4,)
                  |
                  ▼
       self.people[name] = average
                  |
                  ▼
             self.save()
                  |
                  ▼
             📁 JSON FILE
```

---

## 10. 📐 Real Project — Actual Dimensions

Imagine we have 5 face snapshots, each containing 512 values.

| Operation | Shape / Result |
|---|---|
| Input fingerprints | 5 arrays × 512 values |
| `np.stack(fingerprints)` | `(5, 512)` |
| `.mean(axis=0)` | `(512,)` |
| `np.linalg.norm(average)` | One number |
| After normalization | `(512,)` |
| Saved fingerprint | 512 values |

---

## 11. 🧠 Five Things to Remember

1. `np.stack()` combines separate arrays into one 2D array.
2. `mean(axis=0)` averages each feature across all snapshots.
3. `np.linalg.norm()` calculates the vector's length.
4. `/=` normalizes the vector by dividing its values by that length.
5. `self.people[name] = average` stores or replaces the person's fingerprint, and `self.save()` writes the updated database to JSON.

---

## 12. 🎓 Test Your Understanding

Try answering these without looking at the notes:

**Q1.** If we have 7 snapshots and each contains 512 numbers, what is `stacked.shape`?

**Q2.** Why does `mean(axis=0)` change the shape from `(7, 512)` to `(512,)`?

**Q3.** Does normalization change the shape or only the values?

**Q4.** If Talha is already in the dictionary and we register him again, what happens to his old fingerprint?

**Q5.** What is the responsibility of `self.save()`?

---

### ⭐ The Main Idea

**Many snapshots → One average fingerprint → Normalize → Store → Save to JSON.**

The goal is not to memorize these lines. The goal is to understand what each step does and why the program needs it.
## 📐 NumPy Array Shapes — Quick Notes

- `(4,)` → 1D array containing 4 values. No rows/columns dimension.
- `(1, 4)` → 2D array containing 1 row and 4 columns.
- `(3, 4)` → 2D array containing 3 rows and 4 columns.

### How `np.stack()` works

Each input fingerprint becomes one row.

```python
np.stack([
    [1, 2, 3, 4],
    [2, 4, 6, 8],
    [3, 6, 9, 12]
])
```

Shape: `(3, 4)` → 3 fingerprints × 4 values each.

### How `mean(axis=0)` works

`axis=0` averages down the rows, column by column.

```text
(3, 4) → mean(axis=0) → (4,)
```

Result: `[2, 4, 6, 8]`

**Remember:** The comma in `(4,)` indicates a one-dimensional shape. It does not mean an empty column.