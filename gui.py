
from tkinter import *
import tkinter as tk
import os
from PIL import ImageTk, Image



def camera():
    os.system("python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model")
def facerecognition():
    os.system("python facerecognition.py")

def facedataset():
    os.system("python 1facedataset.py")
def facetraining():
    os.system("python 2facetraining.py")
root = Tk()
width_value=1024
height_value=768
root.geometry("%dx%d+0+0"% (width_value,height_value))


image2 = Image.open('D:/FACE22/bg.png')
#image2.show()
image1 = ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=image1)
background_label.image1=image1
background_label.place(x=0, y=0, height=height_value, width=width_value)




root.title("app")
b=Button(root,text="dataset",command=facedataset)
b.pack(side="top", padx=0, pady=10)
b2=Button(root,text="training",command=facetraining)
b2.pack(side="top", padx=0, pady=10)
b3=Button(root,text="recognition",command=facerecognition)
b3.pack(side="top", padx=0, pady=10)
b4=Button(root,text="object",command=camera)
b4.pack(side="top", padx=0, pady=10)
root.mainloop()
