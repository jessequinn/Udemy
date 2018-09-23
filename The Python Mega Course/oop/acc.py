class Account:

    def __init__(self, fp='balance.txt'):
        self.fp = fp
        self._balance = 0

        self._balance_()

    def _balance_(self):
        with open(self.fp, 'r') as file:
            self._balance = int(file.read())

    def withdraw(self, amount):
        self._balance = self._balance - amount

    def deposit(self, amount):
        self._balance = self._balance + amount

    def commit(self):
        with open(self.fp, 'w') as file:
            file.write(str(self._balance))

    def __str__(self):
        return "Current Balance: {} ".format(self._balance)


class Checking(Account):
    """
    This class generates checking account objects
    """

    type = "checking"

    def __init__(self, fee=0):
        self.fee = fee

        Account.__init__(self, fp='checkings.txt')

    def transfer(self, amount, type):
        if type == '-':
            self._balance = self._balance - amount - self.fee
        else:
            self._balance = self._balance + amount - self.fee

if __name__ == '__main__':
    acc = Account()
    chk = Checking()

    print(acc)

    acc.withdraw(100)
    acc.commit()

    print(acc)

    acc.deposit(300)
    acc.commit()

    print(acc)
    print(chk)

    chk.transfer(100, '+')
    chk.commit()

    print(chk)

    print(chk.__doc__)
