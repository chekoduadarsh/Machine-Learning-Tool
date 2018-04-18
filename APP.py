from tkinter import *

import matplotlib.pyplot as plt

import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

class app:
    def __init__(self,root):
        self.f=Frame(root,height=300,width=600)
        self.f.propagate(0)
        self.f.pack()
        self.but1 = Button(self.f, text="Simple Linear Regression",command=self.SLR)
        self.but2 = Button(self.f, text="Multiple Linear Regression",command=self.MLR)
        self.but3 = Button(self.f, text="decision tree regression",command=self.DTR)
        self.but4 = Button(self.f, text="Convolutional Neural Network",command=self.CNN)
        self.l1=Label(self.f,text="Regression", font=('Areal',-30,'bold underline'))
        self.l2=Label(self.f,text="Neural Networks", font=('Areal',-30,'bold underline'))
        self.l3=Label(self.f,text="By,\nAdarsh C  \n V - 1.2 \n support: chekodu.adarsh@gmail.com")
        self.l1.place(x=200,y=10)
        self.but1.place(x=10,y=50)
        self.but2.place(x=200,y=50)
        self.but3.place(x=400,y=50)
        self.l2.place(x=150,y=90)
        self.but4.place(x=10,y=130)
        self.l3.place(x=350,y=230)
    def SLR(self):
        os.system('pyhton MLTF.py')
    def MLR(self):
        os.system('pyhton MLTMLR.py')
    def DTR(self):
        os.system('pyhton MLTDTR.py')
    def CNN(self):
        os.system('pyhton MLTCNN.py')

        
root=Tk()
root.title("Machine Learning Toolbox")
#sec.geometry("400x250")
mb=app(root)
root.mainloop()
