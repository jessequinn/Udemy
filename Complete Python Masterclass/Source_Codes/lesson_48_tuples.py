# t = "a", "b", "c"
# print(t)
#
# print("a", "b", "c")
# print(("a", "b", "c"))
welcome = "Welcome to my Nightmare", "Alice Cooper", 1975
bad = "Bad Company" "Bad Company", 1974
budgie = "Nightflight", "Budgie", 1981
imelda = "More Mayhem", "Imilda May", 2011
metallica = "Ride the Lightning", "Metallica", 1984

# to correct a tuple
imelda = imelda[0], "Imilda May", imelda[2]
print(imelda)


metallica2 = ["Ride the Lightning", "Metallica", 1984]
metallica2[0] = "Master of puppets"
print(metallica2)

title, artist, year = imelda
print(title)
print(artist)
print(year)
