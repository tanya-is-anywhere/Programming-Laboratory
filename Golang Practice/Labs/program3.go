package main

import (
	"fmt"
	"math"
)

func main() {

	const pi = math.Pi
	const exp = math.E
	// площадь круга
	var r = 5.0
	S := pi * r * r
	// значение функции экспоненты
	x := 3.0
	y := math.Pow(exp, x)

	fmt.Println(S, y)
}
