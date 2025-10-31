
class UserAccount():
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.__password = password
    def set_password(self, new_password):
        self.__password = new_password
    def check_password(self, password):
        return self.__password == password

UserAccount1 = UserAccount("Владислав", "vlad_fm_@yandex.ru", "e8rg48rg4")

UserAccount1.set_password("125869")
print(UserAccount.check_password("125869"))


