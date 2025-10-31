package main

import (
	"fmt"
	"unsafe"
)

func main() {
	var int_, float_, complex_, byte_, string_, bool_ = 5, 28.5, 4 + 5i, 'B', "world is imaginary", true

	fmt.Println("Размер типа int:", unsafe.Sizeof(int_))
	fmt.Println("Размер типа float64:", unsafe.Sizeof(float_))
	fmt.Println("Размер типа complex128:", unsafe.Sizeof(complex_))
	fmt.Println("Размер типа byte:", unsafe.Sizeof(byte_))
	fmt.Println("Размер типа string:", unsafe.Sizeof(string_))
	fmt.Println("Размер типа bool:", unsafe.Sizeof(bool_))
}
