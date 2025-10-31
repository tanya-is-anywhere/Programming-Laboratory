package main

import (
	"fmt"
	"math"
)

// создание интерфейса Форма
type Form interface {
	areaCircle() float64
	areaSquare() float64
}
type Data struct {
	A float64 //одна из сторон прямоугольника
	B float64
	R float64 //радиус круга
}

func (d Data) areaCircle() {
	fmt.Printf("Площадь круга равна: %.2f\n", d.R*d.R*math.Pi)
}
func (d Data) areaSquaree() {
	fmt.Printf("Площадь квадрата равна: %.2f\n", d.A*d.B)
}
func WhatareaC(d Data) {
	d.areaCircle()
}
func WhatareaS(d Data) {
	d.areaSquaree()
}

func main() {
	var Data1 = Data{A: 2, B: 5, R: 8}
	WhatareaC(Data1)
	WhatareaS(Data1)
}
