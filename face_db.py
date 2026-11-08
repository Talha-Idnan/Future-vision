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
    # config.db means it is going in config file and looking for db file and using its path
    def __init__(self, path=config.DB_FILE):
        self.path = path
        # make empty dict 
                # name -> numpy array (the averaged fingerprint)
        self.people = {}
        # use load function and fill people dict witha data from json file if exists 
        self.load()

    # ---------- saving and loading ----------

    def load(self):
        """Read the saved file from disk (if it exists)."""
        # cheak if file exxist if not then it will exit whole load function due to return keword
        if not os.path.exists(self.path):
            return
        # open the file read dat A INTO DATA as a python dict
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # JSON stores plain lists, so convert them back into numpy arrays
        # self.people = {name: np.array(vec, dtype=np.float32) for name, vec in data.items()}
    # search for name and its value in data then convert it in to np array then whole in to people dict with name as key and np array as vale
        for name, vec in data.items():
            self.people[name] = np.array(vec, dtype=np.float32)
            # prints basic statment obout how many people are loaded and their names
        print(f"Loaded {len(self.people)} known people: {', '.join(self.people) or 'none'}")

    def save(self):
        """Write everything to disk."""
        # it cheaks weater faces folder exizsts or not if not then it will creat one
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # numpy arrays cannot go into JSON directly, so convert them to lists with help of .tolist ,np convert in to numpy array ok
        
#       or this both re same   data = {}
#     for name, vec in self.people.items():
#       data[name] = vec.tolist()
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
        # mkae all np array in to one combined big array withoud doing opereartions i mean joning opreations
        stacked = np.stack(fingerprints)# shape: (num_snapshots, 512)
        # get average of all snapshots column and make it in average variable           
        average = stacked.mean(axis=0)  # shape: (512,)   
        # divides  array by its length so that it is normalized       
        average /= np.linalg.norm(average)         # shrink back to length 1
        self.people[name] = average
        self.save()

# we will fidst use load function on same obkect so tahrt we get all anmes from ourt 
# json file then cheak modify and use save function to save it permanently rewrite kind off
    def delete_person(self, name):
        """Remove a person. Returns True if they existed."""
        if name not in self.people:
            return False
            # del removes an entry from the dictionary
        del self.people[name]
        self.save()
        return True

    def names(self):
        return list(self.people.keys())

    # ---------- recognizing ----------

# Identify a person by comparing their fingerprint with everyone in the database.
# If the database is empty, return Unknown.
# Calculate the Euclidean distance for each stored fingerprint and keep the closest match.
# If the smallest distance is within MATCH_THRESHOLD, return the person's name.
# Otherwise, return Unknown.

    def identify(self, fingerprint):
        """
        Compare one fingerprint with everyone we know.

        Returns (name, distance):
          - name is the closest person, or "Unknown" if nobody is close enough
          - distance is how far apart the fingerprints are (smaller = more similar)
          - distance is None if the database is empty
        """
        # to cheak if there is no one in the data base then it will return unknown and none as distance
        if not self.people:
            return "Unknown", None
        # variable we have to find for best march to our 
        best_name, best_dist = None, None
        for name, stored in self.people.items():
            # Euclidean distance: straight-line distance between two points in 512-D space

            # it is complicaed math opreation which give distand between 
            dist = float(np.linalg.norm(fingerprint - stored))
            if best_dist is None or dist < best_dist:
                best_name, best_dist = name, dist

        if best_dist <= config.MATCH_THRESHOLD:
            return best_name, best_dist
        return "Unknown", best_dist
