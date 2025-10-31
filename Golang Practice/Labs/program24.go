package main

import "fmt"

// создание структуры Студент
type Student struct {
	Name         string
	BornYear     int
	Curse        int
	AverageScore float32
}

func (s Student) methodAGE() {
	var year = 2025
	var age = year - s.BornYear
	fmt.Printf("Студенту по имени %s %d курса на данный момент %d лет.\n", s.Name, s.Curse, age)
}
func (s Student) methodSTATUS() {
	if s.AverageScore == 5.0 {
		fmt.Printf("Студент по имени %s %d является отличником.\n", s.Name, s.Curse)
	} else if s.AverageScore < 5.0 && 3.0 < s.AverageScore {
		fmt.Printf("Студент по имени %s %d является хорошистом.\n", s.Name, s.Curse)
	} else if s.AverageScore < 3.0 && 2.0 < s.AverageScore {
		fmt.Printf("Студент по имени %s %d является троечником.\n", s.Name, s.Curse)
	}

}
func main() {

	/* функции для работы со структурой
	1) создать студента
	2) вывести данные студента
	3) изменить данные студента */
	s1 := Student{Name: "Eve", BornYear: 2006, Curse: 2, AverageScore: 4.9}
	fmt.Printf("Вы вывели данные студента по имени %s.\nСейчас учится на %d курсе, год рождения %d, средний балл: %.2f\n", s1.Name, s1.Curse, s1.BornYear, s1.AverageScore)
	s1.BornYear = 2007
	fmt.Printf("Вы вывели данные студента по имени %s.\nСейчас учится на %d курсе, год рождения %d, средний балл: %.2f\n", s1.Name, s1.Curse, s1.BornYear, s1.AverageScore)
	// тестирование методов
	s1.methodAGE()
	s1.methodSTATUS()

}
