import os.path
import sqlite3


class Database:

    def __init__(self, db='books.db'):
        self.db = db

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.BASE_DIR, self.db)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS book (id INTEGER PRIMARY KEY, title TEXT, author TEXT, year INTEGER, isbn INTEGER)")
            conn.commit()

    def insert(self, title, author, year, isbn):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO book VALUES (NULL,?,?,?,?)", (title, author, year, isbn))
            conn.commit()

    def view(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM book")
            rows = cur.fetchall()

        return rows

    def search(self, title="", author="", year="", isbn=""):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM book WHERE title=? OR author=? OR year=? OR isbn=?", (title, author, year, isbn))
            rows = cur.fetchall()

        return rows

    def delete(self, id):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM book WHERE id=?", (id,))
            conn.commit()

    def update(self, id, title, author, year, isbn):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE book SET title=?, author=?, year=?, isbn=? WHERE id=?", (title, author, year, isbn, id))
            conn.commit()
