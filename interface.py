import tkinter
from tkinter import *
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
from nueral_network import Layer_Dense, Activation_ReLU, Activation_Softmax, dense1, activation1, dense2, activation2

try:
    dense1.weights = np.load('w1.npy')
    dense1.biases = np.load('b1.npy')
    dense2.weights = np.load('w2.npy')
    dense2.biases = np.load('b2.npy')
    print("OK")
except FileNotFoundError:
    print("NOT OK")

root = Tk()
root.title("Neural Network Numbers Recognition")

message = Label(root, text="Draw a number between 0 and 9")
message.pack(padx=0, pady=0)

canvas = Canvas(root, width=280, height=280, bg="black")
canvas.pack(padx=20, pady=20)

result_label = Label(root, text="Prediction: None", font=("Arial", 18), fg="black")
result_label.pack(pady=10)

draw_image = Image.new('L', (280, 280), 0)
draw_canvas = ImageDraw.Draw(draw_image)

last_x, last_y = None, None

def paint(event):
    global last_x, last_y
    if last_x is None:
        last_x, last_y = event.x, event.y
        return
    canvas.create_line(last_x, last_y, event.x, event.y, fill="white", width=15, capstyle=ROUND, smooth=TRUE)
    draw_canvas.line([last_x, last_y, event.x, event.y], fill=255, width=15)
    last_x, last_y = event.x, event.y

def reset(event=None):
    global last_x, last_y
    last_x, last_y = None, None

def clear(event=None):
    canvas.delete("all")
    draw_canvas.rectangle([0, 0, 280, 280], fill=0)
    reset()
    result_label.config(text="Prediction: None")

def predict():
    img = draw_image.copy()
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    img_array = np.array(img)
    ys, xs = np.where(img_array > 30)

    if len(xs) == 0:
        result_label.config(text="Prediction: None")
        return

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    w, h = img.size
    max_size = max(w, h)

    padded = Image.new('L', (max_size, max_size), 0)
    padded.paste(img, ((max_size - w) // 2, (max_size - h) // 2))

    new_size = 20
    img_resized = padded.resize((new_size, new_size), Image.Resampling.LANCZOS)

    final = Image.new('L', (28, 28), 0)
    final.paste(img_resized, (4, 4))

    img_array = np.array(final).astype('float32') / 255.0
    img_flattened = img_array.reshape(1, 784)

    dense1.forward(img_flattened)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    prediction = np.argmax(activation2.output)
    confidence = activation2.output[0][prediction] * 100
    result_label.config(text=f"Prediction: {prediction} ({confidence:.1f}%)")

canvas.bind('<B1-Motion>', paint)
canvas.bind('<ButtonRelease-1>', reset)

clear_button = tkinter.Button(root, text="Clear", command=clear, font=("New Roman", 12), width=15, height=2)
predict_button = tkinter.Button(root, text="Predict", command=predict, font=("New Roman", 12), width=15, height=2)

clear_button.pack(padx=5, pady=5)
predict_button.pack(padx=10, pady=10)
mainloop()