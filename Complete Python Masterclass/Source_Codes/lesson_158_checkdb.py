import sqlite3

conn = sqlite3.connect("lesson_155_contacts.sqlite")

name = input("Please enter a name to search for ")

# for row in conn.execute("SELECT * FROM contacts WHERE name = ?", (name,)):
for row in conn.execute("SELECT * FROM contacts WHERE name LIKE ?", (name,)):
    print(row)

conn.close()
