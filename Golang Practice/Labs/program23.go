package main

import (
	"fmt"
)

func multiply(a int) {
	a = a * 2
}
func multiply1(a *int) {
	*a = *a * 2
}
func main() {
	age := 20
	fmt.Println("Исходное число:", age)
	multiply(age)
	fmt.Println("Изменение исходной переменной при подаче значения:", age) // значение не изменяется
	multiply1(&age)
	fmt.Println("Изменение исходной переменной при подаче указателя:", age) // значение изменяется

}
