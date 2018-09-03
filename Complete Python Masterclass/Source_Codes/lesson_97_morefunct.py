try:
    import tkinter
except ImportError:  # python 2
    import Tkinter as tkinter


def parabola(x):
    y = x * x / 100
    return y


def draw_axes(canvas):
    canvas.update()
    x_origin = canvas.winfo_width() / 2
    y_origin = canvas.winfo_height() / 2
    canvas.configure(scrollregion=(-x_origin, -y_origin, x_origin, y_origin))
    canvas.create_line(-x_origin, 0, x_origin, 0, fill="black")
    canvas.create_line(0, y_origin, 0, -y_origin, fill="black")


def plot(canvas, x, y):
    canvas.create_line(x, y, x + 1, y + 1, fill="red")


# GUI setup
mainWindow = tkinter.Tk()
mainWindow.update_idletasks()  # need to accurately retrieve screen_width and _height

# mainWindow centering
screen_width = mainWindow.winfo_screenwidth()
screen_height = mainWindow.winfo_screenheight()
size = tuple(int(_) for _ in mainWindow.geometry().split('+')[0].split('x'))
x = screen_width/2 - size[0]/2
y = screen_height/2 - size[1]/2

mainWindow.title("Parabola")
# mainWindow.geometry("640x480")
mainWindow.geometry("+%d+%d" % (x, y))

# canvas = tkinter.Canvas(mainWindow, width=640, height=480)
canvas = tkinter.Canvas(mainWindow)

canvas.grid(row=0, column=0)

draw_axes(canvas)

for x in range(-100, 100):
    y = parabola(x)
    plot(canvas, x, -y)

mainWindow.mainloop()
