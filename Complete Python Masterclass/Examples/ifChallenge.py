
name = input("What is your name? ")
age = int(input("How old are you, {0}? ".format(name)))

if 18 <= age < 31:
    print("Welcome to your holiday, {0}!".format(name))
else:
    print("Sorry, you have no holiday {0}.".format(name))