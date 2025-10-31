package main

import "fmt"

// создание структуры Банковский счёт
type BankAccount struct {
	Number  int
	Balance float64
}

func (s *BankAccount) withdrawMoney(summa float64) { //снять деньги
	if s.Balance >= summa {
		s.Balance -= summa
		fmt.Printf("Вы сняли со счёта под номером %d %.2f рублей. Ваш баланс составляет %.2f\n", s.Number, summa, s.Balance)
	} else {
		fmt.Printf("Операция со счёта под номером %d отклонена. Причина: недостаточно средств для снятия.\n", s.Number)
	}
}
func (s *BankAccount) topUp(summa float64) { //пополнить баланс
	s.Balance += summa
	fmt.Printf("Счёт под номером %d пополнен на %.2f рублей. Ваш баланс составляет %.2f\n", s.Number, summa, s.Balance)
}
func (s *BankAccount) checkBalance() { //проверить баланс
	fmt.Printf("Ваш баланс составляет %.2f\n", s.Balance)
}

func main() {
	var account1 = BankAccount{8788932130, 50000}
	// функции для работы со структурой Банковский счёт
	account1.checkBalance()
	account1.withdrawMoney(500)
	account1.checkBalance()
	account1.topUp(1500)
	account1.checkBalance()
}
