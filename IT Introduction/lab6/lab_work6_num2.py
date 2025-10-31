
class Vehicle():
    def __init__(self, make, model):
        self.make = make
        self.model = model
    def get_info(self):
        return f"Марка: {self.make} Модель: {self.model}"

class Car(Vehicle):
    def __init__(self, make, model, fuel_type):
        super().__init__(make, model)
        self.fuel_type = fuel_type
    def get_info(self):
        return f"Марка: {self.make}, Модель: {self.model}, Тип топлива: {self.fuel_type}"

car1 = Car("Chevrolet", "Chevrolet Colorado", "бензин")

print(car1.get_info())

