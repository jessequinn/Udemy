# cities = ["Adelaide", "Alice Springs", "Darwin", "Melbourne", "Sydney"]

# fname = "lesson_65_cities.txt"

# with open(fname, 'w') as city_file:
#     for city in cities:
#         print(city, file=city_file)

# cities = []

# with open(fname, 'r') as city_file:
#     for city in city_file:
#         cities.append(city.strip('\n'))

# print(cities)
# for city in cities:
#     print(city)

fname = "lesson_65_imelda3.txt"

imelda = "More Mayhem", "Imelda MAy", "2011", (
    (1, "Pulling the Rug"), (2, "Psycho"), (3, "Mayhem"), (4, "Kentish Town Waltz"))

with open(fname, 'w') as imelda_file:
    print(imelda, file=imelda_file)

with open(fname, 'r') as imelda_file:
    contents = imelda_file.readline()

imelda = eval(contents)

# print(imelda)
title, artist, year, tracks = imelda
print(title)
print(artist)
print(year)
print(tracks[0][1])
