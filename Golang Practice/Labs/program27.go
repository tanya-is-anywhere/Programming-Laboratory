package main

import (
	"fmt"
)

// создание интерфейса Транспорт
type transport interface {
	movement()
	stop()
}
type truck struct {
	speed           int
	brakingDistance float64
	loadCapacity    int //в тоннах
}
type bicycle struct {
	speed           int
	brakingDistance float64
}

func (t truck) movement() {
	fmt.Printf("Грузовик с грузоподъёмностью %d тонн начинает двигаться со скоростью %d км/ч\n", t.loadCapacity, t.speed)
}
func (t truck) stop() {
	fmt.Printf("Грузовик начинает останавливаться. Тормозной путь занял %.2f метров.\n", t.brakingDistance)
}
func truckMove(t truck) {
	t.movement()
}
func truckStop(t truck) {
	t.stop()
}

func (b bicycle) movement() {
	fmt.Printf("Велосипед начинает двигаться со скоростью %d км/ч\n", b.speed)
}
func (b bicycle) stop() {
	fmt.Printf("Велосипед начинает останавливаться. Тормозной путь занял %.2f метров.\n", b.brakingDistance)
}
func bicycleMove(b bicycle) {
	b.movement()
}
func bicycleStop(b bicycle) {
	b.stop()
}
func main() {
	var truck1 = truck{60, 18.88, 25}
	truckMove(truck1)
	truckStop(truck1)
	var bicycle1 = bicycle{15, 1.18}
	bicycleMove(bicycle1)
	bicycleStop(bicycle1)
}
