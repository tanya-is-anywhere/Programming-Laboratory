package main

import "fmt"

func main() {
	// создание структуры Двигатель
	type Engine struct {
		Type   string
		Power  int
		Volume float64
	}
	// создание структуры Автомобиль
	type Car struct {
		Brand       string
		Model       string
		ReleaseYear int
		Mileage     int
		Engine      Engine
	}
	// создать двигатель
	engine1 := Engine{"Бензиновый", 150, 2.5}
	// создать автомобиль
	car1 := Car{"Toyota",
		"Camry",
		2020,
		25000,
		engine1}
	// вывод информации об автомобиле
	fmt.Println("Марка:", car1.Brand)
	fmt.Println("Модель:", car1.Model)
	fmt.Println("Год выпуска:", car1.ReleaseYear)
	fmt.Println("Пробег:", car1.Mileage, "км")
	fmt.Println("Двигатель:")
	fmt.Println("  Тип:", car1.Engine.Type)
	fmt.Println("  Мощность:", car1.Engine.Power, "л.с.")
	fmt.Println("  Объем:", car1.Engine.Volume, "л")
	// увеличить пробег авто
	var adding_m = 150
	car1.Mileage += adding_m
	fmt.Printf("Пробег машины %s %s увеличен на %d км. Теперь пробег составляет %d км", car1.Brand, car1.Model, adding_m, car1.Mileage)
}
