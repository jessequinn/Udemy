import os.path
import sqlite3


def connect():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "books.db")
    print(os.getcwd())
    with sqlite3.connect(db_path) as conn:
        # conn = sqlite3.connect("./books.db")
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS book (id INTEGER PRIMARY KEY, title TEXT, author TEXT, year INTEGER, isbn INTEGER)")
        conn.commit()
        # conn.close()


def insert(title, author, year, isbn):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "books.db")
    with sqlite3.connect(db_path) as conn:
        # conn = sqlite3.connect("./books.db")
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO book VALUES (NULL,?,?,?,?)", (title, author, year, isbn))
        conn.commit()
        # conn.close()


def view():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "books.db")
    with sqlite3.connect(db_path) as conn:
        # conn = sqlite3.connect("./books.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM book")
        rows = cur.fetchall()
        # conn.close()

    return rows


def search(title="", author="", year="", isbn=""):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "books.db")
    with sqlite3.connect(db_path) as conn:
        # conn = sqlite3.connect("./books.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM book WHERE title=? OR author=? OR year=? OR isbn=?", (title, author, year, isbn))
        rows = cur.fetchall()
        # conn.close()

    return rows


def delete(id):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "books.db")
    with sqlite3.connect(db_path) as conn:
        # conn = sqlite3.connect("./books.db")
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM book WHERE id=?", (id,))
        conn.commit()
        # conn.close()


def update(id, title, author, year, isbn):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "books.db")
    with sqlite3.connect(db_path) as conn:
        # conn = sqlite3.connect("./books.db")
        cur = conn.cursor()
        cur.execute(
            "UPDATE book SET title=?, author=?, year=?, isbn=? WHERE id=?", (title, author, year, isbn, id))
        conn.commit()
        # conn.close()

# if __name__ == '__main__':
# connect()
# insert('The sea', 'John Tablet', 1918, 913123132)
# update(1, 'Test', 'Jese', 1930, 32434324324)
# print(search(author='John Tablet'))
# print(view())
