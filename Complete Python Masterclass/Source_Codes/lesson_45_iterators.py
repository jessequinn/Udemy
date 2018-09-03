# Create a list of items (you may use either strings or numbers in the list),
# then create an iterator using the iter() function.
#
# Use a for loop to loop "n" times, where n is the number of items in your list.
# Each time round the loop, use next() on your list to print the next item.
#
# hint: use the len() function rather than counting the number of items in the list.

my_list = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

my_iterator = iter(my_list)

for i in range(0, len(my_list)):
    next_item = next(my_iterator)
    print(next_item)

for i in my_list:
    print(i)

string = "1234567890"

for char in string:
    print(char)

my_iterator = iter(string)
print(next(my_iterator))