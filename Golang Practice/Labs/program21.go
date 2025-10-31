package main

import "fmt"

func thelongest(stroki ...string) string {
	longest_ := ""
	for i := 0; i < len(stroki); i++ {
		if len(longest_) < len(stroki[i]) {
			longest_ = stroki[i]
		}

	}
	return longest_
}
func main() {
	// демонстрация работы функции
	fmt.Println(thelongest("jfern", "red", "blueberry"))
}
