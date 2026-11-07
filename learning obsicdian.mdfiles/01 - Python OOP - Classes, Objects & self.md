# Python OOP --- Classes, Objects & `self`

> \[!abstract\] In one sentence A **class** defines what an object can
> do and store; an **object** is one instance, and `self` lets its
> methods work with that particular object.

## 1. Class vs. object

-   **Class:** a blueprint for creating objects.
-   **Object (instance):** one individual thing created from a class.

``` python
class Student:
    pass

student1 = Student()
student2 = Student()
```

`Student` is the class. `student1` and `student2` refer to two separate
objects.

``` mermaid
flowchart TD
    A[Student class] --> B[student1 object]
    A --> C[student2 object]
```

## 2. `__init__()` --- initialize an object

`__init__()` is a special method Python calls automatically after a new
object has been created. It sets up the object's starting data.

``` python
class Student:
    def __init__(self, name):
        self.name = name

student = Student("Ali")
print(student.name)  # Ali
```

When `Student("Ali")` is evaluated, Python creates an object and calls
its initializer with that object and `"Ali"`.

## 3. What does `self` mean?

-   `self` refers to the **current object**.
-   It is the first parameter of an instance method by convention.
-   When you call `student.introduce()`, Python automatically passes
    `student` as the method's first argument.

``` python
class Student:
    def __init__(self, name):
        self.name = name

    def introduce(self):
        print("My name is", self.name)

student = Student("Ali")
student.introduce()
```

These calls are conceptually equivalent:

``` python
student.introduce()
Student.introduce(student)
```

The second form makes the automatically passed object visible. You
normally use the first form.

## 4. Instance attributes

An **instance attribute** is data stored on one particular object.

``` python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

student1 = Student("Ali", 20)
student2 = Student("Sara", 19)

print(student1.name)  # Ali
print(student2.name)  # Sara
```

-   `self.name` and `self.age` create/store attributes on the current
    object.
-   Dot notation accesses them: `student1.name`.
-   Each object has its own values.

### Local variable vs. instance attribute

``` python
def example(self):
    message = "Hello"       # local variable
    self.name = "Ali"       # instance attribute
```

-   `message` is local to that method call.
-   `self.name` belongs to the object and can be accessed later through
    that object, e.g. `student.name`.

## 5. Applying `self` to `FaceDatabase`

``` python
class FaceDatabase:
    def __init__(self, path):
        self.path = path
        self.people = {}
```

Creating an object:

``` python
CD = FaceDatabase("faces/database.json")
```

During this call, `self` refers to the newly created object that `CD`
will refer to. Conceptually, the assignments have the effect of:

``` python
CD.path = "faces/database.json"
CD.people = {}
```

This is a **conceptual explanation**, not a literal text replacement
Python performs. Python passes the object as `self` when it calls the
initializer.

Access the attributes with dot notation:

``` python
print(CD.path)
print(CD.people)
CD.people["Ali"] = [1, 2, 3]
```

## 6. Two objects, separate instance data

``` python
CD1 = FaceDatabase("database1.json")
CD2 = FaceDatabase("database2.json")

CD1.people["Ali"] = [1, 2, 3]

print(CD1.people)  # {'Ali': [1, 2, 3]}
print(CD2.people)  # {}
```

Each initialization runs `self.people = {}`, creating a new dictionary
for that object. Changing `CD1.people` does not directly change
`CD2.people`.

> \[!warning\] Important project detail Your actual
> `FaceDatabase.__init__()` also calls `self.load()`. If a database file
> already contains saved people, that object's dictionary may be
> populated from the file. So `CD2.people` is `{}` only if its file has
> no saved entries (or does not exist).

## Quick recap

  Concept              Meaning
  -------------------- -----------------------------------------
  Class                Blueprint for objects
  Object / instance    One object created from a class
  `__init__()`         Automatically initializes a new object
  `self`               The current object
  `self.name`          An instance attribute
  `object.attribute`   Dot notation to access an attribute
  Local variable       Temporary name inside a function/method

## Practice --- try before checking

1.  In `car1 = Car()`, which part is the class and which part is the
    object reference?
2.  When does `__init__()` run?
3.  In `self.path = path`, what does `self` refer to?
4.  If two objects each run `self.people = {}`, do they share the same
    dictionary?
5.  What is the difference between `message = "Hi"` and
    `self.message = "Hi"` inside a method?

```{=html}
<details>
```
```{=html}
<summary>
```
Answers
```{=html}
</summary>
```
1.  `Car` is the class; `car1` refers to the created object.
2.  Automatically during object initialization, when a new object is
    created.
3.  The object whose initializer/method is currently running.
4.  No. Each object gets its own dictionary.
5.  `message` is a local variable; `self.message` is stored on the
    object.

```{=html}
</details>
```

------------------------------------------------------------------------

**Next note:** \[\[02 - FaceDatabase, JSON & Memory\]\]
