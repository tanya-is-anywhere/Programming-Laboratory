package main

import (
	"fmt"
	"strings"
)

func main() {
	text := "Isn't it surprising how quickly the seasons change from winter to spring?"
	map_ := make(map[string]int)
	text_ := strings.Split(text, " ")
	for _, i := range text_ {
		map_[i] = strings.Count(text, i)
	}
	fmt.Println(map_)
}
