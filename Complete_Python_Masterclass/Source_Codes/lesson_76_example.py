import random

help(random.randint)


# built in functions
# print(dir()
# print(dir(__builtins__))

# for m in dir(__builtins__):
#     print(m)

import shelve

# print(dir())
# print()

for m in dir(shelve.Shelf):
    print(m)