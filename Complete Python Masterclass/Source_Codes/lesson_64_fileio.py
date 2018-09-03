# import os

fname = "lesson_64_sample.txt"

# try:
#     jabber = open(fname, 'r')
#     # jabber = open("C:\\Documents and Settings\\tim\\My Documents\\sample.txt", 'r')

# except IOError:
#     print("Could not read file: ", fname)

# for line in jabber:
#     if "jabberwock" in line.lower():
#         print(line, end='')

# jabber.close()

# with open(fname, 'r') as jabber:
#     for line in jabber:
#         if "JAB" in line.upper():
#             print(line, end='')


# with open(fname, 'r') as jabber:
#     line = jabber.readline()
#     while line:
#         print(line, end='') # end = "" removes newline 
#         line = jabber.readline()

# with open(fname, 'r') as jabber:
#     lines = jabber.readlines()
# print(lines)

# for line in lines:
#     print(line, end='')


# with open(fname, 'r') as jabber:
#     lines = jabber.readlines()
# print(lines)

# for line in lines[::-1]:
#     print(line, end='')

with open(fname, 'r') as jabber:
    lines = jabber.read()

for line in lines[::-1]:
    print(line, end='')
