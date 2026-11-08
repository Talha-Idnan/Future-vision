import json
import os
from os import path

class students:
    file_path = r"G:\future vision\code_practices\students.json"
    def __init__(self,name,age):
        self.name = name
        self.age = age
        self.add_student()

    def load_all(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def add_student(self):
        data = self.load_all()
        data[self.name] = self.age

        with open(self.file_path,"w") as f:
            json.dump(data,f)

    def show_all_students(self):
        print(self.load_all())

a = students("talha", 20)
b = students("ali", 16)
c = students("fatiam", 17)

b.show_all_students()



