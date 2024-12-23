
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def get_radius(self):
        print("Радиус круга: {}".format(self.radius))

    def set_radius(self, new_radius):
        self.radius = new_radius
        print("Новый радиус круга: {}".format(self.radius))


circle1 = Circle(25)
circle1.get_radius()
circle1.set_radius(35)
circle1.get_radius()
