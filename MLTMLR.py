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
        self.e2 = Entry(self.f, bd = 5)
        self.l2 = Label(self.f, text="Test and Train split ratio")
        self.but1 = Button(self.f, text="Save",command=self.split)
        self.l3 = Label(self.f, text="Select CSV file from browse")
        self.e4 = Entry(self.f, bd = 5)
        self.l4 = Label(self.f, text="Digrees")
        self.but2 = Button(self.f, text="Save",command=self.dig)
        self.br = Button(self.f, text="Browse",command=self.browse)
        self.run = Button(self.f, text="RUN",command=self.RUN)
        #self.but.pack()
        #self.but1.pack()
        #self.br.pack()
        self.l1.place(x=10,y=10)
        self.e1.place(x=160,y=10)
        self.but.place(x=330,y=10)
        #self.l2.place(x=10,y=40)
        #self.e2.place(x=160,y=40)
        self.l4.place(x=10,y=80)
        self.e4.place(x=160,y=80)
        self.but2.place(x=330,y=80)
        #self.but1.place(x=330,y=40)
        self.l3.place(x=10,y=120)
        self.br.place(x=100,y=160)
        self.run.place(x=320,y=220)
        
    def var(self):
        self.variable=self.e1.get()
        print(self.variable)
        
    def split(self):
        self.ratio=self.e2.get()
        print(self.variable)
        print(self.ratio)
        
    def dig(self):
        print(self.variable)
        self.digr=self.e4.get()
        print(self.digr)
        
    def browse(self):
        filename = filedialog.askopenfilename()
        self.path= filename
        print (self.path)
        
        #pathlabel.config(text=filename)
    def RUN(self):
        di = int(self.digr)
        
        # Importing the dataset
        dataset = pd.read_csv(self.path)
        X = dataset.iloc[:, 1:2].values
        y = dataset.iloc[:, 2].values
        
# Fitting Linear Regression to the dataset
        from sklearn.linear_model import LinearRegression
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
        from sklearn.preprocessing import PolynomialFeatures
        poly_reg = PolynomialFeatures(di)
        X_poly = poly_reg.fit_transform(X)
        poly_reg.fit(X_poly, y)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
        #plt.scatter(X, y, color = 'red')
        #plt.plot(X, lin_reg.predict(X), color = 'blue')
        #plt.title('(Linear Regression)')
        #plt.show()
        #plt.scatter(X, y, color = 'red')
        #plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
        #plt.title('(Polynomial Regression)')
        #plt.show()
        X_grid = np.arange(min(X), max(X), 0.1)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color = 'red')
        plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
        plt.title('(Polynomial Regression)higher resolution and smoother curve')
        plt.show()
        


root=Tk()
root.title("Machine Learning Toolbox")
#sec.geometry("400x250")
mb=app(root)
root.mainloop()
