package main

import (
	"fmt"
)

func main() {
	num := 5 // задается число, для которого нужно получить результаты умножения
	for i := 1; i <= 10; i++ {

		fmt.Println("При умножении числа", num, "на", i, "получим", i*num)

	}
}
