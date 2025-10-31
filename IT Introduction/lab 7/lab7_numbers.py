class Employee:
    def __init__(self, name, id):
        self.name = name
        self.id = id
    def get_info(self):
        print(f"Имя: {self.name}\nID: {self.id}")

class Manager(Employee):
    def __init__(self, name, id, department):
        Employee.__init__(self, name, id)
        self.department = department
    def manage_project(self):
        print(f"Менеджер {self.name} (id:{self.id}) департамента {self.department} управляет проектом")

class Technician(Employee):
    def __init__(self, name, id, specialisation):
        Employee.__init__(self, name, id)
        self.specialisation = specialisation
    def perform_maintenance(self):
        print(f"Технический специалист {self.name} (id:{self.id}) по специализации {self.specialisation} выполняет техническое обслуживание")


class TechManager(Manager, Technician):
    def __init__(self, name, id, specialisation, department):
        Manager.__init__(self, name, id, department)
        Technician.__init__(self, name, id, specialisation)
        self.team = []

    def add_employee(self, employee):
        self.team.append(employee)

    def get_team_info(self):
        return [employee.get_info() for employee in self.team]


worker01 = Employee("Alice", "222q")
worker01.get_info()
worker02 = Manager("Alice", "222q", "safety")
worker02.get_info()
worker02.manage_project()
worker03 = Technician("Alice", "222q", "safety")
worker03.get_info()
worker03.perform_maintenance()

worker1 = TechManager("Artur","84845", "we", "are")
worker1.get_info()

tech1 = Technician("Ronald", "232b0", "электрик")
tech2 = Technician("Mamed", "332b0", "инженер")
tech3 = Technician("Oleg", "432b0", "слесарь")

worker1.add_employee(tech1)
worker1.add_employee(tech2)
worker1.add_employee(tech3)

worker1.get_team_info()
