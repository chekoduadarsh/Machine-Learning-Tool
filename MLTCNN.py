from tkinter import *

import matplotlib.pyplot as plt

import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split


class app:
    def __init__(self,root):
        self.f=Frame(root,height=300,width=400)
        self.f.propagate(0)
        self.f.pack()
        self.e1 = Entry(self.f, bd =5)
        self.l1 = Label(self.f, text="Type the path to trainingset")
        self.but = Button(self.f, text="Save",command=self.dat)
        self.e22 = Entry(self.f, bd =5)
        self.l22 = Label(self.f, text="Type the path to testset")
        self.but22 = Button(self.f, text="Save",command=self.dat1)
        self.e2 = Entry(self.f, bd = 5)
        self.l2 = Label(self.f, text="Path to new image")
        self.but1 = Button(self.f, text="Save",command=self.new)
        self.l3 = Label(self.f, text="Select CSV file from browse")
        self.e4 = Entry(self.f, bd = 5)
        self.l4 = Label(self.f, text="STEPS TO EPOCH")
        self.but2 = Button(self.f, text="Save",command=self.steps)
        self.e5 = Entry(self.f, bd = 5)
        self.l5 = Label(self.f, text="EPOCH")
        self.but3 = Button(self.f, text="Save",command=self.epoch)
        #self.br = Button(self.f, text="Browse",command=self.browse)
        self.run = Button(self.f, text="RUN",command=self.RUN)
        #self.but.pack()
        #self.but1.pack()
        #self.br.pack()
        self.l1.place(x=10,y=10)
        self.e1.place(x=160,y=10)
        self.but.place(x=330,y=10)
        self.l22.place(x=10,y=40)
        self.e22.place(x=160,y=40)
        self.but22.place(x=330,y=40)
        self.l2.place(x=10,y=70)
        self.e2.place(x=160,y=70)
        self.but1.place(x=330,y=70)
        self.l4.place(x=10,y=100)
        self.e4.place(x=160,y=100)
        self.but2.place(x=330,y=100)
        self.l5.place(x=10,y=130)
        self.e5.place(x=160,y=130)
        self.but3.place(x=330,y=130)
        #self.l3.place(x=10,y=120)
        #self.br.place(x=100,y=160)
        self.run.place(x=320,y=220)

    def steps(self):
        self.step=self.e4.get()
        print(self.step)
        
    def epoch(self):
        self.epoc=self.e5.get()
        print(self.epoc)
        
    def dat(self):
        self.data=self.e1.get()
        print(self.data)

    def dat1(self):
        self.data1=self.e1.get()
        print(self.data1)
        
    def new(self):
        self.ne=self.e2.get()
        print(self.ne)
        print(self.data)
        
    def RUN(self):
        epoch=int(self.epoc)
        steps=int(self.step)
        path=self.data
        path1=self.data1
        # Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.layers import Flatten
        from keras.layers import Dense

        classifier = Sequential()
   #Convolution
        classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
    #Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
        classifier.add(Conv2D(32, (3, 3), activation="relu"))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Flattening
        classifier.add(Flatten())
    # =Full connection
        classifier.add(Dense(activation = 'relu', units = 128))
        classifier.add(Dense(activation = 'sigmoid', units = 1))
    # Compiling
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    #  Fitting the CNN to the images
        from keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale = 1./255)
        training_set = train_datagen.flow_from_directory(path,
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')

        test_set = test_datagen.flow_from_directory(path1,
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

        pro=Label(self.f,text="Processing please wait")
        self.pro.place(x=10,y=280)
        classifier.fit_generator(training_set,
                             steps_per_epoch = steps,
                             epochs = epoch,
                             validation_data = test_set,
                             validation_steps = 63)
        if result[0][0] == 1:
            pro=Label(self.f,text="Type 1 detected")
            self.pro.place(x=10,y=280)
        else:
            pro=Label(self.f,text="Type 2 detected")
            self.pro.place(x=10,y=280)
        

root=Tk()
root.title("Machine Learning Toolbox")
#sec.geometry("400x250")
mb=app(root)
root.mainloop()

