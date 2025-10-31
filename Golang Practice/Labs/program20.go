package main

import (
	"fmt"
	"sort"
)

func Searching(i int, s []int) { // функция поиска индекса элемента
	for b, val := range s {
		if val == i {
			fmt.Printf("Индекс элемента со значением %d: %d\n", i, b)
		}
	}

}
func Sort_(a int, s []int) { // функция сортировки
	if a == 0 { // сортировка по возрастанию
		sort.IntSlice.Sort(s)
		fmt.Println("Результат сортировки (по возрастанию):", s)
	} else if a == 1 { // сортировка по убыванию
		sort.Reverse(sort.IntSlice(s))
		fmt.Println("Результат сортировки (по убыванию):", s)
	}

}
func Filter_(b int, s []int) { // функция фильтрации
	filterS := []int{}
	if b == 0 { // оставить только ЧЕТНЫЕ элементы

		for _, value := range s {
			if value%2 == 0 {
				filterS = append(filterS, value) // Добавляем элемент в результирующий срез
			}

		}

	} else if b == 1 { // оставить только НЕЧЕТНЫЕ элементы
		filterS := []int{}
		for _, value := range s {
			if value%2 != 0 {
				filterS = append(filterS, value) // Добавляем элемент в результирующий срез
			}

		}
	}
	fmt.Println("Результат фильтрации:", filterS)
}
func main() {
	// создать срез
	s := []int{252, 75, 85, 4}
	// Демонтрация работы функций
	Searching(75, s)
	Sort_(1, s)
	Filter_(0, s)

}
