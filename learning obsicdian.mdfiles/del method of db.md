# `delete_person()` Method

**Purpose:** Deletes a person from the database and saves the updated data permanently.

```python
def delete_person(self, name):
    if name not in self.people:
        return False

    self.people.pop(name)
    self.save()
    return True
```

### How it works

1. `if name not in self.people:` — Checks whether the person exists as a dictionary key.
2. `return False` — Stops if the person doesn't exist.
3. `self.people.pop(name)` — Removes the person's key-value entry from the dictionary in RAM.
4. `self.save()` — Saves the updated dictionary into the JSON file.
5. `return True` — Confirms successful deletion.

### Important concepts

- `self.people` accesses the dictionary stored inside the current object.
- `pop(name)` removes a dictionary entry. `del self.people[name]` does the same.
- `return` immediately exits the method.
- `load()` reads JSON → Python dictionary.
- `save()` writes Python dictionary → JSON file.

**Remember:** Delete changes data in RAM; save makes that change permanent in the JSON file.