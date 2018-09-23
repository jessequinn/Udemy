import sqlite3

import psycopg2

# global database selector
_postgres_ = True


def create_table(db=None):
    if _postgres_:
        conn = psycopg2.connect("dbname=new_database user=jessequinn password=`9155` host=localhost port=5432")
    else:
        conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS store (item TEXT, quantity INTEGER, price REAL)")
    conn.commit()
    conn.close()

def alter_table(db=None):
    if _postgres_:
        conn = psycopg2.connect("dbname=new_database user=jessequinn password=`9155` host=localhost port=5432")
    else:
        conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("ALTER TABLE store ADD UNIQUE (item)")
    conn.commit()
    conn.close()


def insert(item, quantity, price, db=None):
    if _postgres_:
        conn = psycopg2.connect("dbname=new_database user=jessequinn password=`9155` host=localhost port=5432")
    else:
        conn = sqlite3.connect(db)
    cur = conn.cursor()
    if _postgres_:
        try:
            cur.execute("INSERT INTO store VALUES (%s,%s,%s)", (item, quantity, price))
        except:
            print('Item exists!')
    else:
        cur.execute("INSERT INTO store VALUES (?,?,?)", (item, quantity, price))
    conn.commit()
    conn.close()


def view(db=None):
    if _postgres_:
        conn = psycopg2.connect("dbname=new_database user=jessequinn password=`9155` host=localhost port=5432")
    else:
        conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT * FROM store")
    rows = cur.fetchall()
    conn.close()
    return rows


def delete(item, db=None):
    if _postgres_:
        conn = psycopg2.connect("dbname=new_database user=jessequinn password=`9155` host=localhost port=5432")
    else:
        conn = sqlite3.connect(db)
    cur = conn.cursor()
    if _postgres_:
        cur.execute("DELETE FROM store WHERE item=%s", (item,))
    else:
        cur.execute("DELETE FROM store WHERE item=?", (item,))
    conn.commit()
    conn.close()


def update(quantity, price, item, db=None):
    if _postgres_:
        conn = psycopg2.connect("dbname=new_database user=jessequinn password=`9155` host=localhost port=5432")
    else:
        conn = sqlite3.connect(db)
    cur = conn.cursor()
    if _postgres_:
        cur.execute("UPDATE store SET quantity=%s, price=%s WHERE item=%s", (quantity, price, item))
    else:
        cur.execute("UPDATE store SET quantity=?, price=? WHERE item=?", (quantity, price, item))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    if not _postgres_:
        data = 'lite.db'
        create_table(data)
        alter_table(data)
        insert('Water Glass', 10, 5.0, data)
        insert('Coffe Cup', 4, 3.0, data)
        delete('Water Glass', data)
        update(9, 5.5, 'Coffe Cup', data)
        print(view(data))
    else:
        create_table()
        alter_table()
        insert('Water Glass', 10, 5.0)
        insert('Coffee Cup', 4, 3.0)
        delete('Water Glass')
        update(9, 5.5, 'Coffee Cup')
        print(view())
