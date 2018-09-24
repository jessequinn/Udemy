import tkinter as tk

class DragDropMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.drag_start_x = 0
        self.drag_start_y = 0
        self.bind("<Button-1>", self.drag_start)
        self.bind("<B1-Motion>", self.drag_motion)

    def drag_start(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def drag_motion(self, event):
        x = self.winfo_x() - self.drag_start_x + event.x
        y = self.winfo_y() - self.drag_start_y + event.y
        self.place(x=x, y=y)


# As always when it comes to mixins, make sure to
# inherit from DragDropMixin FIRST!
class DnDFrame(DragDropMixin, tk.Frame):
    pass


# This wouldn't work:
# class DnDFrame(tk.Frame, DragDropMixin):
#     pass

main = tk.Tk()
notesFrame = DnDFrame(main, bd=4, bg="grey")
notesFrame.place(x=10, y=10)
notes = tk.Text(notesFrame)
notes.pack()

main.mainloop()
