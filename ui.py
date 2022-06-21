import cv2
import numpy as np
from tkinter import *
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
import cv2
import pickle
import numpy as np
from PIL import Image, ImageTk


model = load_model('Model_MAE2.h5')
root = Tk()
root.configure(background="white")

root.title("Dungdeptrai")
root.geometry("250x300")
mainFrame = Frame(root)
mainFrame.place(x=50, y=70)
lmain = Label(mainFrame)
lmain.grid(row=1, column=0)
lmain1 = Label(mainFrame)
lmain1.grid(row=2, column=1)
vid = cv2.VideoCapture('Pose Estimation/test.mp4')

X_DATA_PATH = 'data.pickle'
Y_DATA_PATH = 'Pose Estimation/point.csv'

y_data = pd.read_csv(Y_DATA_PATH)
y_data.head(None)

x_data = pickle.load(open(X_DATA_PATH, 'rb'))
x_data = np.array(x_data, dtype='float32')
x_data /= 255


input_shape = x_data.shape[1:4]
y_data = np.array(y_data, dtype='float')
num_class = y_data.shape[1]
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.15)


def recognize_age():
    while(1):
        ret, img = vid.read()
        img = cv2.resize(img, (250, 250), fx=0, fy=0,
                         interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32')
        img /= 255
        pred = model.predict(img.reshape(1, 250, 250, 3))
        points = pred[:]
        points = points.reshape(-1)

        cv2.line(img, (points[2*0].astype('int'), points[2*0+1].astype('int')),
                 (points[2*1].astype('int'), points[2*1+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*1].astype('int'), points[2*1+1].astype('int')),
                 (points[2*2].astype('int'), points[2*2+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*1].astype('int'), points[2*1+1].astype('int')),
                 (points[2*3].astype('int'), points[2*3+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*3].astype('int'), points[2*3+1].astype('int')),
                 (points[2*4].astype('int'), points[2*4+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*4].astype('int'), points[2*4+1].astype('int')),
                 (points[2*5].astype('int'), points[2*5+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*1].astype('int'), points[2*1+1].astype('int')),
                 (points[2*6].astype('int'), points[2*6+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*6].astype('int'), points[2*6+1].astype('int')),
                 (points[2*7].astype('int'), points[2*7+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*7].astype('int'), points[2*7+1].astype('int')),
                 (points[2*8].astype('int'), points[2*8+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*2].astype('int'), points[2*2+1].astype('int')),
                 (points[2*9].astype('int'), points[2*9+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*9].astype('int'), points[2*9+1].astype('int')),
                 (points[2*10].astype('int'), points[2*10+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*10].astype('int'), points[2*10+1].astype('int')),
                 (points[2*11].astype('int'), points[2*11+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*2].astype('int'), points[2*2+1].astype('int')),
                 (points[2*12].astype('int'), points[2*12+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*12].astype('int'), points[2*12+1].astype('int')),
                 (points[2*13].astype('int'), points[2*13+1].astype('int')), (0, 0, 255), 2)
        cv2.line(img, (points[2*13].astype('int'), points[2*13+1].astype('int')),
                 (points[2*14].astype('int'), points[2*14+1].astype('int')), (0, 0, 255), 2)

        for j in range(15):
            x = points[2*j]
            x = x.astype('int')
            if x < 0:
                x = x*(-1)
            y = points[2*j+1]
            y = y.astype('int')
            if y < 0:
                y = y*(-1)

            cv2.circle(img, (x, y), 2, (255, 0, 0), 2)

        cv2.imshow('Dungdeptrai', cv2.resize(img, (250, 250),
                   fx=0, fy=0, interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(1)

    img = Image.fromarray(img).resize((320, 320))
    img_open = ImageTk.PhotoImage(image=img)
    lmain.imgtk = img_open
    lmain.configure(image=img_open)
    lmain.after(15, recognize_age)


def stop_program():
    quit()


while True:
    cap_btn = Button(root, text='Start', font=("Arial", 14, "bold"),
                     bd=3, bg="green", foreground="black", command=recognize_age)
    cap_btn.place(x=40, y=250)
    cap_btn2 = Button(root, text='Stop', font=("Arial", 14, "bold"),
                      bg="red", foreground="black", command=stop_program)
    cap_btn2.place(x=160, y=250)
    tit = Label(root, text='POSE', bd=3, bg='white',
                fg='blue', font=("Robotic", 20, "bold"))
    tit.place(x=80, y=0)
    tit = Label(root, text='estimation', bd=3, bg='white',
                fg='blue', font=("Robotic", 16, "bold"))
    tit.place(x=65, y=30)
    root.mainloop()
