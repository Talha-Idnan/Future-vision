"""
config.py
---------
All the settings in ONE place.

If the app behaves wrongly, change a number here instead of digging
through main.py. Most tuning happens in MATCH_THRESHOLD.
"""

import os

# ---------------- Camera ----------------
# cmera index 0 means select camera default one we givetit in config so that we came hange between
#  diffrent camera without going in to main
CAMERA_INDEX = 0            # 0 = default webcam (try 1 if you have several cameras)
# you can change resolution to lower if you want better fps 
FRAME_WIDTH = 1280          # higher = sharper picture, but a little slower
FRAME_HEIGHT = 720

# ---------------- Face detection (MTCNN: "where is a face?") ----------------
# model need to be sure obout person at least 90 persent before giving prediction 
FACE_MIN_CONFIDENCE = 0.90  # ignore faces the detector is less than 90% sure about
MIN_FACE_SIZE = 60          # ignore faces smaller than 60 pixels (too blurry to recognize)

# ---------------- Face recognition (InceptionResnetV1: "who is it?") ----------------
RECOGNITION_MODEL = "vggface2"  # pretrained weights that come with facenet_pytorch

# How different two fingerprints can be and still count as the same person.
#   Smaller = stricter  -> fewer wrong names, but more "Unknown"
#   Bigger  = looser    -> more names found, but higher risk of mixing people up
# Start here, then tune using the distance numbers shown on screen (SHOW_DISTANCE).
# This controls how similar two face embeddings must be to count as the same person.
MATCH_THRESHOLD = 0.9

# Voting: the name shown is the most common answer over the last N frames.
# This stops 
# Show the distance number next to each name (useful while tuning, turn off later)
SHOW_DISTANCE = True

# ---------------- Enrollment (adding people) ----------------
MIN_SNAPSHOTS = 3           # how many 'c' presses are needed before you can save a person

# ---------------- Object detection (YOLO: "what else is in the room?") ----------------
YOLO_MODEL_PATH = "yolo26n.pt"
YOLO_CONFIDENCE = 0.50
SHOW_PEOPLE_FROM_YOLO = False  # False = no big body boxes, only face boxes

# ---------------- Storage ----------------
# it means foleder name faces
DB_DIR = "faces"
# joins folder name with file name make full location of file we will use this file many times 
DB_FILE = os.path.join(DB_DIR, "database.json")
# db_file =faces/database.json
