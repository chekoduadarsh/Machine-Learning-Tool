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
        self.l1 = Label(self.f, text="Number of variables")
        self.but = Button(self.f, text="Save",command=self.var)
        self.e2 = Entry(self.f, bd =5)
        self.l2 = Label(self.f, text="Test and Train split ratio")
        self.but1 = Button(self.f, text="Save",command=self.split)
        self.l3 = Label(self.f, text="Select CSV file from browse")
        self.br = Button(self.f, text="Browse",command=self.browse)
        self.run = Button(self.f, text="RUN",command=self.RUN)
        self.run = Button(self.f, text="RUN",command=self.RUN)
        self.but.pack()
        self.but1.pack()
        self.br.pack()
        self.l1.place(x=10,y=10)
        self.e1.place(x=160,y=10)
        self.but.place(x=330,y=10)
        self.l2.place(x=10,y=40)
        self.e2.place(x=160,y=40)
        self.but1.place(x=330,y=40)
        self.l3.place(x=10,y=80)
        self.br.place(x=100,y=120)
        self.run.place(x=320,y=220)
        
    def var(self):
        self.variable=self.e1.get()
        print(self.variable)
        
    def split(self):
        self.ratio=self.e2.get()
        print(self.variable)
        print(self.ratio)
        
    def browse(self):
        filename = filedialog.askopenfilename()
        self.path= filename
        print (self.path)
        
        #pathlabel.config(text=filename)
    def RUN(self):
        rat=float(self.ratio)        
        # Importing the dataset
        dataset = pd.read_csv(self.path)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = rat, random_state = 0)

# Fitting Simple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

# Predicting the Test set results
        y_pred = regressor.predict(X_test)
        import matplotlib.pyplot as plt
# Visualising the Training set results  
        plt.scatter(X_train, y_train, color = 'red')
        plt.plot(X_train, regressor.predict(X_train), color = 'blue')
        plt.title('(Training set)')
        plt.show()

# Visualising the Test set results
        plt.scatter(X_test, y_test, color = 'red')
        plt.plot(X_train, regressor.predict(X_train), color = 'blue')
        plt.title('(Test set)')
        plt.show()

root=Tk()
root.title("Machine Learning Toolbox")
#sec.geometry("400x250")
mb=app(root)
root.mainloop()
