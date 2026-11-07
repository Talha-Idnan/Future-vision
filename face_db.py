"""
face_db.py
----------
A small "album" of known people.

For each person we store ONE averaged fingerprint (a list of 512 numbers),
saved to faces/database.json so it survives restarts.

Main jobs:
  - add_person(name, fingerprints)  -> average the snapshots, save them
  - delete_person(name)             -> remove someone
  - identify(fingerprint)           -> return the closest name, or "Unknown"
"""

import json
import os

import numpy as np

import config


class FaceDatabase:
    def __init__(self, path=config.DB_FILE):
        self.path = path
        self.people = {}        # name -> numpy array (the averaged fingerprint)
        self.load()

    # ---------- saving and loading ----------

    def load(self):
        """Read the saved file from disk (if it exists)."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # JSON stores plain lists, so convert them back into numpy arrays
        self.people = {name: np.array(vec, dtype=np.float32) for name, vec in data.items()}
        print(f"Loaded {len(self.people)} known people: {', '.join(self.people) or 'none'}")

    def save(self):
        """Write everything to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # numpy arrays cannot go into JSON directly, so convert them to lists
        data = {name: vec.tolist() for name, vec in self.people.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ---------- adding and removing ----------

    def add_person(self, name, fingerprints):
        """
        Average several snapshots into one fingerprint and save it.
        Averaging smooths out one bad angle or bad lighting.
        If the name already exists, it is replaced.
        """
        stacked = np.stack(fingerprints)           # shape: (num_snapshots, 512)
        average = stacked.mean(axis=0)             # shape: (512,)
        average /= np.linalg.norm(average)         # shrink back to length 1
        self.people[name] = average
        self.save()

    def delete_person(self, name):
        """Remove a person. Returns True if they existed."""
        if name not in self.people:
            return False
        del self.people[name]
        self.save()
        return True

    def names(self):
        return list(self.people.keys())

    # ---------- recognizing ----------

    def identify(self, fingerprint):
        """
        Compare one fingerprint with everyone we know.

        Returns (name, distance):
          - name is the closest person, or "Unknown" if nobody is close enough
          - distance is how far apart the fingerprints are (smaller = more similar)
          - distance is None if the database is empty
        """
        if not self.people:
            return "Unknown", None

        best_name, best_dist = None, None
        for name, stored in self.people.items():
            # Euclidean distance: straight-line distance between two points in 512-D space
            dist = float(np.linalg.norm(fingerprint - stored))
            if best_dist is None or dist < best_dist:
                best_name, best_dist = name, dist

        if best_dist <= config.MATCH_THRESHOLD:
            return best_name, best_dist
        return "Unknown", best_dist
