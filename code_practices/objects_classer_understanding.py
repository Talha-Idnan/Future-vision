class my_self:
    def __init__(self,name,age):
        self.name = name

    def introduction(self):
        print(f'my name is {self.name} and I am {self.age} years old')

my = my_self('talha',15)
# print(my.name)

my.introduction()