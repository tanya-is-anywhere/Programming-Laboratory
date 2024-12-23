# Лабораторная работа №2: Функции в Python и базовые алгоритмы

# Цель работы: Освоить принципы определения и использования функций в языке программирования Python, понять механизмы передачи аргументов в функции, научиться применять функции для решения практических задач, а также изучить базовые алгоритмические конструкции.

# Задание 1
def greet(name):
    return f"Привет, {name}! Добро пожаловать в программу!"
def square(number):
    return number**2
def max_of_two(x, y):
    if x > y:
        return x
    else:
        return y

# Задание 2
def describe_person(name, age=30):
    print("Имя: ")
    print("Возраст: ")

# Задание 3
def is_prime(n):
    l = 1
    for i in range(2, int(n**0.5)+1):
        if n%i==0:
            l+=1
    if l == 1:
        return True
    else: return False


print(is_prime(13))

