package main

import (
	"fmt"
	"strings"
)

func main() {
	var s = "What a wonderful idea it is to plant trees in our local park!"
	w := strings.Split(s, " ")
	for _, word := range w {
		fmt.Println(word)
	}
}
