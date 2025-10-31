package main

import (
	"fmt"
)

func main() {
	day := 25
	switch day {
	case 1:
		fmt.Println(day, "- Понедельник")
	case 2:
		fmt.Println(day, "- Вторник")
	case 3:
		fmt.Println(day, "- Среда")
	case 4:
		fmt.Println(day, "- Четверг")
	case 5:
		fmt.Println(day, "- Пятница")
	case 6:
		fmt.Println(day, "- Суббота")
	case 7:
		fmt.Println(day, "- Воскресенье")
	default:
		fmt.Println(day, "В неделе только 7 дней")
	}
}
