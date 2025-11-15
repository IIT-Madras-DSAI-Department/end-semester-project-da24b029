import numpy as np
import time
import pandas as pd


def load_csv(path):
    df=pd.read_csv(path)
    return df.values

def accuracy(pred,y):
    return np.mean(pred==y)

def confusion_matrix(pred,y,C=10):
    cm=np.zeros((C,C),dtype=int)
    for t,p in zip(y,pred):
        cm[int(t),int(p)] +=1
    return cm

def precision_macro(pred,y):
    cm=confusion_matrix(pred,y)
    vals=[]
    for c in range(10):
        tp=cm[c,c]
        fp=cm[:,c].sum()-tp
        vals.append(tp/(tp+fp+1e-9))
    return np.mean(vals)

def recall_macro(pred,y):
    cm=confusion_matrix(pred,y)
    vals=[]
    for c in range(10):
        tp=cm[c,c]
        fn=cm[c,:].sum()-tp
        vals.append(tp/(tp+fn+1e-9))
    return np.mean(vals)

def macro_f1(pred,y):
    p= precision_macro(pred,y)
    r= recall_macro(pred,y)
    return 2*p*r/(p+r+1e-9)


class KNN:
    def __init__(self,k=7):
        self.k=k

    def fit(self,X,y):
        start=time.time()
        self.X_train=X
        self.y_train=y
        self.train_time=time.time()-start

    def predict(self,X):
        preds=[]
        for x in X:
            dist=np.linalg.norm(self.X_train-x,axis=1)
            k_idx=np.argsort(dist)[:self.k]
            k_labels=self.y_train[k_idx]
            preds.append(np.bincount(k_labels,minlength=10).argmax())
        return np.array(preds)

def main():
    train=load_csv("MNIST_train.csv")
    test=load_csv("MNIST_validation.csv")

    X_train=train[:,:-1]/255.0
    y_train=train[:,-1].astype(int)

    X_test=test[:,:-1]/255.0
    y_test=test[:,-1].astype(int)

    knn=KNN(k=7)
    knn.fit(X_train,y_train)
    pred=knn.predict(X_test)

    # Metrics
    acc=accuracy(pred,y_test)
    prec=precision_macro(pred,y_test)
    rec=recall_macro(pred,y_test)
    f1=macro_f1(pred,y_test)

    print("\n KNN CLASSIFIER RESULTS ")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Train Time: {knn.train_time:.4f} sec")

if __name__=="__main__":
    main()

