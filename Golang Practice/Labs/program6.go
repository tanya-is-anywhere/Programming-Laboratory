package main

import (
	"fmt"
	"strings"
)

func main() {
	var s = "Hello, World!"
	fmt.Println(len(s))                    //подсчет символов
	fmt.Println(strings.Contains(s, "lo")) //поиск подстроки
	fmt.Println(strings.ToUpper(s))        //изменение регистра на верхний
	fmt.Println(strings.ToLower(s))        // изменение регистра на нижний
	fmt.Println(s)
}
