class Employee:
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name")
        self.id = kwargs.get("id")
    def get_info(self):
        print(f"Имя: {self.name}\nID: {self.id}")

class Manager(Employee):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.department = kwargs.get("department")
    def manage_project(self):
        print(f"Менеджер {self.name} (id:{self.id}) департамента {self.department} управляет проектом")

class Technician(Employee):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.specialisation = kwargs.get("specialisation")
    def perform_maintenance(self):
        print(f"Технический специалист {self.name} (id:{self.id}) по специализации {self.specialisation} выполняет техническое обслуживание")


class TechManager(Manager, Technician):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.team = list(kwargs.get("team"))

    def add_employee(self, employee):
        self.team.append(employee)

    def get_team_info(self):
        return [employee.get_info() for employee in self.team]


worker01 = Employee(name="Alice", id="222q")
worker01.get_info()
worker02 = Manager(name="Alice", id="222q", department="safety")
worker02.get_info()
worker02.manage_project()
worker03 = Technician(name="Alice", id="222q", specialisation="safety")
worker03.get_info()
worker03.perform_maintenance()

worker1 = TechManager(name="Artur", id="84845", department="we", specialisation="are", team=[])
worker1.get_info()

tech1 = Technician(name="Ronald", id="232b0", specialisation="электрик")
tech2 = Technician(name="Mamed", id="332b0", specialisation="инженер")
tech3 = Technician(name="Oleg", id="432b0", specialisation="слесарь")

worker1.add_employee(tech1)
worker1.add_employee(tech2)
worker1.add_employee(tech3)

worker1.get_team_info()
