import tkinter
from tkinter import *
from PIL import Image, ImageOps, ImageGrab
import numpy as np

root = Tk()
message = Label(root, text="Draw a number between 0 and 9")
message.pack(padx=0, pady=0)
canvas = Canvas(root, width=600, height=500,bg="black")
canvas.pack(padx=20, pady=20)
root.title("Neural Network Numbers Recognition")



last_x,last_y = None, None

def paint(event):
    python_black = "#ffffff"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill=python_black, outline=python_black)

def clear():
    canvas.delete("all")

def predict():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    img = ImageGrab.grab().crop((x, y, x1, y1))

    img = img.convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    img_array = np.array(img) / 255.0

    img_flattened = img_array.reshape(1, 784)

canvas.bind('<B1-Motion>', paint)


clear_button = tkinter.Button(root,
                   text="Clear",
                   command=clear,
                   activebackground="black",
                   activeforeground="white",
                   anchor="center",
                   bd=3,
                   bg="white",
                   cursor="hand2",
                   disabledforeground="gray",
                   fg="black",
                   font=("Arial", 12),
                   height=2,
                   highlightbackground="black",
                   highlightcolor="green",
                   highlightthickness=2,
                   justify="center",
                   overrelief="raised",
                   padx=10,
                   pady=5,
                   width=15,
                   wraplength=100)

predict_button = tkinter.Button(root,
                   text="Predict",
                   command=predict,
                   activebackground="black",
                   activeforeground="white",
                   anchor="center",
                   bd=3,
                   bg="white",
                   cursor="hand2",
                   disabledforeground="gray",
                   fg="black",
                   font=("Arial", 12),
                   height=2,
                   highlightbackground="black",
                   highlightcolor="green",
                   highlightthickness=2,
                   justify="center",
                   overrelief="raised",
                   padx=10,
                   pady=5,
                   width=15,
                   wraplength=100)

clear_button.pack(padx=5, pady=5)
predict_button.pack(padx=10, pady=10)
mainloop()
