package main

import (
	"fmt"
)

func main() {
	var days = 366
	var year = 2024
	if days == 366 && year%4 == 0 {
		fmt.Println("Год является високосным.")
	} else {
		fmt.Println("Год не является високосным.")
	}

}
