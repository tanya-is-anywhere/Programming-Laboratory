package main

// программа для поиска простых чисел
import (
	"fmt"
)

func main() {
	for num := 1; num < 100; num++ {
		i := 1
		cnt := 0
		for i < 100 {
			if num%i == 0 {
				cnt++
			}
			i++
		}
		if cnt == 2 {
			fmt.Println(num)
		}

	}
}
