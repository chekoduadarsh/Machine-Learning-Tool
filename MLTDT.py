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
        self.l2.place(x=10,y=40)
        self.e2.place(x=160,y=40)
        #self.l4.place(x=10,y=80)
        #self.e4.place(x=160,y=80)
        #self.but2.place(x=330,y=80)
        self.but1.place(x=330,y=40)
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
        rat=float(self.ratio)
        dataset = pd.read_csv(self.path)
        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = rat, random_state = 0)

# Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)

# Predicting the Test set results
        y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Decision Tree Classification (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()

# Visualising the Test set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Decision Tree Classification (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()

root=Tk()
root.title("Machine Learning Toolbox")
#sec.geometry("400x250")
mb=app(root)
root.mainloop()



