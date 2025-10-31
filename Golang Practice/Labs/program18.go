package main

import "fmt"

func main() {
	// создание структуры Студент
	type Student struct {
		Name         string
		Age          int
		Curse        int
		AverageScore float32
	}
	/* функции для работы со структурой
	1) создать студента
	2) вывести данные студента
	3) изменить данные студента */
	s1 := Student{Name: "Eve", Age: 19, Curse: 2, AverageScore: 4.9}
	fmt.Printf("Вы вывели данные студента по имени %s.\nСейчас учится на %d курсе, возраст %d, средний балл: %.2f\n", s1.Name, s1.Curse, s1.Age, s1.AverageScore)
	s1.Age = 20
	fmt.Printf("Вы вывели данные студента по имени %s.\nСейчас учится на %d курсе, возраст %d, средний балл: %.2f\n", s1.Name, s1.Curse, s1.Age, s1.AverageScore)
}
