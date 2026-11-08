class FaceDatabase:

    def __init__(self):
        self.people = {
            "Ali": [0.1, 0.2],
            "Ahmed": [0.3, 0.4],
            "Sara": [0.5, 0.6]
        }

    def save(self):
        print("Database saved!")

    def delete_person(self, name):

        if name not in self.people:
            return False
        self.people.pop(name)
        self.save()