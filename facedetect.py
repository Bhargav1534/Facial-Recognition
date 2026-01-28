import json
import numpy as np
import os

class LandmarkFaceRecognizer:
    def __init__(self, db_file="face_landmarks_db.json", threshold=0.4):
        self.db_file = db_file
        self.threshold = threshold
        self.database = []

        # Load existing database if it exists
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                data = json.load(f)
                # Convert vectors to numpy arrays
                for entry in data:
                    entry["vector"] = np.array(entry["vector"], dtype=np.float32)
                    self.database.append(entry)

    # ------------------ Distance Metrics ------------------
    @staticmethod
    def euclidean_distance(vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # ------------------ Identify Face ------------------
    def identify_face(self, new_vector):
        """
        Returns the index and distance of closest match or None if unknown
        """
        if len(self.database) == 0:
            return None, None

        min_dist = float("inf")
        match_idx = None

        for i, entry in enumerate(self.database):
            dist = self.euclidean_distance(new_vector, entry["vector"])
            if dist < min_dist:
                min_dist = dist
                match_idx = i

        if min_dist < self.threshold:
            return match_idx, min_dist
        else:
            return None, min_dist

    # ------------------ Add New Face ------------------
    def add_face(self, face_vector, label=None):
        """
        Add a new face vector to the database.
        Optional label for identification (e.g., person's name)
        """
        if label is None:
            label = f"Person_{len(self.database)}"

        entry = {
            "label": label,
            "vector": face_vector.tolist()
        }
        self.database.append(entry)
        self._save_db()

    # ------------------ Save Database ------------------
    def _save_db(self):
        data_to_save = [{"label": e["label"], "vector": e["vector"]} for e in self.database]
        with open(self.db_file, "w") as f:
            json.dump(data_to_save, f, indent=2)
