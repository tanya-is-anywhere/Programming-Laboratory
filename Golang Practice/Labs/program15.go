package main

import (
	"fmt"
)

func main() {
	a := 12
	b := 14
	operator := "//"

	if operator == "+" {
		fmt.Println(a + b)
	} else if operator == "-" {
		fmt.Println(a - b)
	} else if operator == "*" {
		fmt.Println(a * b)
	} else if operator == "/" {
		fmt.Println(a / b)
	} else {
		fmt.Println("Введён неверный оператор. Попробуйте снова.")
	}

}
