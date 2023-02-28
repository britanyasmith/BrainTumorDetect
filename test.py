class Animal(object):
     def __init__(self, speed, is_mammal):
          self.speed = speed
          self.is_mammal = is_mammal

class Cat(Animal):
     def __init__(self, is_hungry):
          super().__init__(10, True)
          self.is_hungry = is_hungry

barry = Cat(True)
print(f"speed: {barry.speed}")
print(f"is a mammal: {barry.is_mammal}")
print(f"feed the cat?: {barry.is_hungry}")