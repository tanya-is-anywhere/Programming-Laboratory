# Лабораторная работа №3:  Работа с файлами в Python: открытие, чтение, запись, работа с исключениями

def example1(reading):
    if reading == 1:
        print("Чтение всего файла")
        with open('../introIT/example', 'r') as file:
            content = file.read()
            print(content)
        print()
    elif reading == 2:
        print("Построчное чтение")
        with open('../introIT/example', 'r') as file:
            for line in file:
                print(line)
        print()

def example2(type=1):
    if type == 1:
        try:
            with open("example3", "a") as f:
                user_input = input("Добавьте новое содержимое: ")
                f.write(user_input)
        except FileNotFoundError:
            print("Данный файл не существует")
    elif type == 2:
        try:
            with open("../introIT/example2", "w") as f:
                user_input = input("Создайте новое содержимое: ")
                f.write(user_input)
        except FileNotFoundError:
            print("Данный файл не существует")

def example3(reading):
    try:
        if reading == 1:
            with open('../introIT/example', 'r') as file:
                print("Чтение всего файла")
                content = file.read()
                print(content)
            print()
        elif reading == 2:
            with open('../introIT/example', 'r') as file:
                print("Построчное чтение")
                for line in file:
                    print(line)
            print()
    except FileNotFoundError:
        print("Файл не найден")



choice = input("Введите задание, которое хотите проверить (1,2 или 3)")
n = int(input("Выберите номер команды (1 или 2) "))

if choice == 1:
    example1(n)
elif choice == 2:
    example2(n)
elif choice == 3:
    example3(n)

