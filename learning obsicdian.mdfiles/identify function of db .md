## identify() Method

**Purpose:** Identify a person by comparing their new fingerprint with everyone stored in the database.

- `if not self.people`: If the database is empty, return `("Unknown", None)`.
- `.items()`: Iterate through every person's name and stored fingerprint.
- `np.linalg.norm()`: Calculate the Euclidean distance between the new and stored fingerprints.
- `best_name` and `best_dist`: Keep track of the closest person and smallest distance.
- `MATCH_THRESHOLD`: Check whether the closest match is close enough.
- If distance is within the threshold, return `(name, distance)`.
- Otherwise, return `("Unknown", distance)`.

**Remember:** Smaller distance means more similar fingerprints. The threshold decides whether the closest person is accepted as a match.
### Euclidean Distance

Euclidean distance measures the straight-line distance between two points (or vectors).

In `identify()`, we use it to compare an incoming face fingerprint with stored fingerprints.

- Smaller distance = more similar fingerprints.
- Distance of 0 = identical fingerprints.

**Formula:**

\[
d=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

In our code:

```python
dist = float(np.linalg.norm(fingerprint - stored))
```

`fingerprint - stored` calculates the difference between corresponding values, and `np.linalg.norm()` calculates the Euclidean distance.