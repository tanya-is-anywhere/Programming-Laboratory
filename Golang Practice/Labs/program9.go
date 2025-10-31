package main

import (
	"fmt"
	"slices" // импорт пакета для работы со срезами
)

func main() {
	mySlice := []string{"One", "Two", "Three", "Four"} // создаём пустой срез
	m := 1                                             // обозначаем индекс элемента, который нужно удалить
	mySlice = slices.Delete(mySlice, m, m+1)           // удаляем элемент с помощью функции Delete
	fmt.Println(mySlice)                               // выводим каждый элемент на новой строке

}
