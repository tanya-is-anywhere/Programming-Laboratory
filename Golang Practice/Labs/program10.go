package main

import (
	"fmt"
)

func main() {
	studentsMarks := make(map[string]int)
	studentsMarks["Михайлова Дарья"] = 5 // добавление элементов
	studentsMarks["Семенов Владимир"] = 5
	studentsMarks["Максимова Карина"] = 4
	studentsMarks["Сергеев Александр"] = 4
	fmt.Println(studentsMarks)
	var searching = "Семенов Владимир" // поиск результата
	fmt.Println(studentsMarks[searching])
	delete(studentsMarks, "Сергеев Александр") // удаление элемента
	fmt.Println("Измененная карта: ", studentsMarks)
}
