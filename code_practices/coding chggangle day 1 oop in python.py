class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        # TODO: Print the student's name and age
        print(f'my name is {self.name} and i am {self.age} yers old ')

    def birthday(self):
        # TODO: Increase the student's age by 1
        self.age +=1
        


# Create two students
student1 = Student("Ali", 20)
student2 = Student("Ahmed", 22)

# Introduce both students
student1.introduce()
student2.introduce()

# Ali celebrates his birthday
student1.birthday()

# Introduce both again
student1.introduce()
student2.introduce()