import shelve

fname = "lesson_72_book.db"

books = shelve.open(fname)

books["recipes"] = {"blt": ["bacon", "lettuce", "tomato", "bread"],
                    "beans_on_toast": ["beans", "bread"],
                    "scrambled eggs": ["eggs", "butter", "milk"],
                    "soup": ["tin of soup"],
                    "pasta": ["pasta", "cheese"]}

books["maintenance"] = {"stuck": ["oil"],
                        "loose": ["gaffer tape"]}

# print(books["recipes"])
print(books["recipes"]["scrambled eggs"])

print(books["maintenance"]["loose"])
books.close()
