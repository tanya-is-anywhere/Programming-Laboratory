package main

import (
	"fmt"
)

func change(a, b int) (int, int) {
	aptr := &a
	bptr := &b
	a, b = *bptr, *aptr
	return a, b
}
func main() {
	// демонстрация работы функции
	one, two := change(1, 2)
	fmt.Println(one, two)
}
