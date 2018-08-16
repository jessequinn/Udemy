# dictionaries
fruit = {
    "orange": "a sweet, orange, citrus frtui",
    "apple": "good for making cider",
    "lemon": "a sour, yellow citrus fruit",
    "grape": "a small, sweet fruit growing in bunches",
    "lime": " a sour, green citruis fruit"
}

# print(fruit)
# print(fruit["lemon"])
fruit["pear"] = "an odd shaped apple"
# print(fruit)
fruit["lime"] = "great with tequila"
# print(fruit)
# del fruit["lemon"]
# fruit.clear()
# print(fruit)
# print(fruit["tomato"])
# print(fruit)
while True:
    dict_key = input("Please enter a fruit: ")
    if dict_key == "quit":
        break
    if dict_key in fruit:
        description = fruit.get(dict_key)
        print(description)
    else:
        print("we don't have a " + dict_key)



#motorbike example

# bike = {"make": "Honda", "model": "250 dream", "colour": "red", "engine_size": 250}
# print(bike["engine_size"])
# print(bike["colour"])
