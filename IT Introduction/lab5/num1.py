
class Book:
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year
    def get_info(self):
        print("Название книги: {}, Автор: {}, Год издания: {}".format(self.title,self.author,self.year))

first = Book("Гарри Поттер и философский камень", "Джоан Роулинг", 1997)

first.get_info()