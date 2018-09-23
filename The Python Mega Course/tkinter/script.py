from tkinter import *

window = Tk()


def km_to_miles():
    print(e1_value.get())
    miles = float(e1_value.get()) * 1.6
    t1.insert(END, miles)


def kg_to_other():
    print(e2_value.get())
    grams = float(e2_value.get()) * 1000
    pounds = float(e2_value.get()) * 2.20462
    ounces = float(e2_value.get()) * 35.274

    t3.delete("1.0", END)
    t3.insert(END, grams)
    t4.delete("1.0", END)
    t4.insert(END, pounds)
    t5.delete("1.0", END)
    t5.insert(END, ounces)


b1 = Button(window, text="Execute", command=km_to_miles)
# b1.pack()
b1.grid(row=0, column=0)

e1_value = StringVar()
e1 = Entry(window, textvariable=e1_value)
e1.grid(row=0, column=1)

t1 = Text(window, height=1, width=20)
t1.grid(row=0, column=2)

l1 = Label(window, text='Kg')
l1.grid(row=1, column=0)

t3 = Text(window, height=1, width=20)
t3.grid(row=2, column=1)
t4 = Text(window, height=1, width=20)
t4.grid(row=2, column=2)
t5 = Text(window, height=1, width=20)
t5.grid(row=2, column=3)


e2_value = StringVar()
e2 = Entry(window, textvariable=e2_value)
e2.grid(row=1, column=2)

b2 = Button(window, text="Convert", command=kg_to_other)
b2.grid(row=1, column=3)

window.mainloop()
