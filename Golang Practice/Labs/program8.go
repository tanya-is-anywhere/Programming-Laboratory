package main

import (
	"fmt"
)

func main() {
	mySlice := []string{}                                 // создаём пустой срез
	mySlice_new := append(mySlice, "One", "Two", "Three") // добавляем элементы динамически
	fmt.Println("Вывод элементов среза: ")                // выводим каждый элемент на новой строке
	for _, i := range mySlice_new {
		fmt.Println(i)
	}
}
