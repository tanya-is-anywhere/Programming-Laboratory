package main

import (
	"fmt"
	"time"
)

func main() {
	now := time.Now()
	fmt.Println("Привет, меня зовут Татьяна. Сегодня", now.Format("02.01.2006"))
}
