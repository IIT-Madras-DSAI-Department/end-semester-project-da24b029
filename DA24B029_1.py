import numpy as np
import pandas as pd
import time
from scipy.special import softmax
from numpy.random import default_rng

rng = default_rng(seed=42)     
np.random.seed(0)

#data load
def load_csv(path):
    df=pd.read_csv(path)
    return df.values


#metrics
def accuracy(pred,y):
    return np.mean(pred==y)

def confusion_matrix(pred,y,C=10):
    cm=np.zeros((C,C),dtype=int)
    for t,p in zip(y,pred):
        cm[int(t),int(p)] +=1
    return cm

def precision_macro(pred,y):
    cm = confusion_matrix(pred,y)
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
        tp = cm[c,c]
        fn = cm[c,:].sum()-tp
        vals.append(tp/(tp+fn+1e-9))
    return np.mean(vals)

def macro_f1(pred,y):
    p=precision_macro(pred,y)
    r=recall_macro(pred,y)
    return 2*p*r/(p+r+1e-9)

# Random Forest 
class DecisionStump:
    def fit(self,X,y,features):
        best_err=np.inf
        for f in features:
            thr=np.median(X[:,f])
            left=X[:,f] <=thr
            right=~left

            left_label=np.bincount(y[left],minlength=10).argmax() if left.any() else 0
            right_label=np.bincount(y[right],minlength=10).argmax() if right.any() else 0

            preds=np.where(left,left_label,right_label)
            err=np.sum(preds != y)

            if err<best_err:
                best_err=err
                self.feature=f
                self.threshold=thr
                self.left_label=left_label
                self.right_label=right_label

    def predict(self,X):
        left = X[:,self.feature] <= self.threshold
        return np.where(left,self.left_label,self.right_label)

class SimpleRandomForest:
    def __init__(self,n_trees=20,max_features=100):
        self.n_trees=n_trees
        self.max_features=max_features

    def fit(self,X,y):
        start=time.time()
        n,d=X.shape
        self.trees=[]
        for _ in range(self.n_trees):
            idx=np.random.choice(n,n,replace=True)
            Xb,yb=X[idx], y[idx]
            features=np.random.choice(d,min(self.max_features,d),replace=False)
            stump=DecisionStump()
            stump.fit(Xb,yb,features)
            self.trees.append(stump)
        self.train_time=time.time() - start

    def predict(self,X):
        all_preds=np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_samples)
        final=[]
        for i in range(all_preds.shape[1]):
            final.append(np.bincount(all_preds[:, i], minlength=10).argmax())
        return np.array(final)


#XGBoost-like(multiclass gradient boosting)
class RegStump:
    def fit(self,X,y_residual,features,n_thresh=20):
        n=X.shape[0]
        best_err=1e18

        for f in features:
            col=X[:,f]

            thresholds=np.random.choice(col,size=min(n_thresh,len(col)),replace=False)

            for thr in thresholds:
                left=col <= thr
                right= ~left

                if not left.any() or not right.any():
                    continue

                lpred=y_residual[left].mean()
                rpred=y_residual[right].mean()

                preds=np.where(left,lpred,rpred)
                err=np.sum((y_residual - preds)**2)

                if err < best_err:
                    best_err=err
                    self.feature=f
                    self.threshold=thr
                    self.left_pred=lpred
                    self.right_pred=rpred

    def predict(self,X):
        left=X[:,self.feature] <= self.threshold
        return np.where(left,self.left_pred,self.right_pred)

class XGBoostLike:
    def __init__(self,n_iters=30,lr=0.3,max_features=100):
        self.n_iters=n_iters
        self.lr=lr
        self.max_features=max_features
        self.trees=[]  

    def fit(self,X,y):
        start=time.time()
        n,d=X.shape
        K=10
        F=np.zeros((n,K))
        #one hot labels
        Y=np.zeros((n,K))
        Y[np.arange(n),y]=1

        for it in range(self.n_iters):
            probs=softmax(F,axis=1)  # (n, K)
            residuals=Y-probs    

            stumps_k=[]
            for k in range(K):
                features=np.random.choice(d,min(self.max_features,d),replace=False)
                stump=RegStump()
                stump.fit(X,residuals[:,k],features,n_thresh=8)
                # predict residuals
                pred_r=stump.predict(X)
                F[:,k] +=self.lr*pred_r
                stumps_k.append(stump)
            self.trees.append(stumps_k)

        self.train_time=time.time() - start

    def predict(self,X):
        n=X.shape[0]
        K=10
        F=np.zeros((n,K))
        for stumps_k in self.trees:
            for k, stump in enumerate(stumps_k):
                F[:,k] += self.lr*stump.predict(X)
        probs=softmax(F,axis=1)
        return np.argmax(probs,axis=1)

# Logistic regression(Softmax Regression)
class SoftmaxRegression:
    def __init__(self,lr=0.05,epochs=60):
        self.lr=lr
        self.epochs=epochs

    def fit(self,X,y):
        start=time.time()
        m,n=X.shape
        K=10
        Y=np.zeros((m,K))
        Y[np.arange(m),y]=1
        self.W=np.zeros((n,K))
        for _ in range(self.epochs):
            logits=X @ self.W
            probs=softmax(logits,axis=1)
            grad=(X.T @ (probs-Y))/m
            self.W -= self.lr*grad
        self.train_time=time.time() - start

    def predict(self,X):
        logits=X @ self.W
        probs=softmax(logits,axis=1)
        return np.argmax(probs,axis=1)

# KNN

class KNN:
    def __init__(self,k=3):
        self.k=k

    def fit(self,X,y):
        start=time.time()
        self.X_train=X
        self.y_train=y
        self.train_time=time.time() - start

    def predict(self,X):
        preds=[]
        for x in X:
            dist=np.linalg.norm(self.X_train - x,axis=1)
            k_idx=np.argsort(dist)[:self.k]
            k_labels=self.y_train[k_idx]
            preds.append(np.bincount(k_labels,minlength=10).argmax())
        return np.array(preds)


#now run all models in order and print results

def run_all():
    train=load_csv("MNIST_train.csv")
    test=load_csv("MNIST_validation.csv")

    X_train=train[:,:-1]/255.0
    y_train=train[:,-1].astype(int)
    X_test=test[:,:-1]/255.0
    y_test=test[:,-1].astype(int)

    results = []

    # 1) Random Forest
    rf=SimpleRandomForest(n_trees=60,max_features=200)
    rf.fit(X_train,y_train)
    rp=rf.predict(X_test)
    acc=accuracy(rp,y_test)
    prec=precision_macro(rp,y_test)
    rec=recall_macro(rp,y_test)
    f1=macro_f1(rp,y_test)
    print("\n RANDOM FOREST Results")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Train Time: {rf.train_time:.2f} sec")
    results.append(("RandomForest", acc, prec, rec, f1, rf.train_time))

    # 2) XGBoost-like
    xgb=XGBoostLike(n_iters=50,lr=0.20,max_features=200)
    xgb.fit(X_train,y_train)
    xp=xgb.predict(X_test)
    acc=accuracy(xp,y_test)
    prec=precision_macro(xp,y_test)
    rec=recall_macro(xp,y_test)
    f1=macro_f1(xp,y_test)
    print("\n XGBOOST-LIKE results")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Train Time: {xgb.train_time:.2f} sec")
    results.append(("XGBoostLike", acc, prec, rec, f1, xgb.train_time))

    # 3) Softmax Regression
    soft=SoftmaxRegression(lr=0.1,epochs=120)
    soft.fit(X_train, y_train)
    sp=soft.predict(X_test)
    acc=accuracy(sp,y_test)
    prec=precision_macro(sp,y_test)
    rec=recall_macro(sp,y_test)
    f1=macro_f1(sp,y_test)
    print("\n SOFTMAX REGRESSION results ")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Train Time: {soft.train_time:.2f} sec")
    results.append(("SoftmaxReg", acc, prec, rec, f1, soft.train_time))

    # 4) KNN
    knn=KNN(k=7)
    knn.fit(X_train,y_train)
    kp=knn.predict(X_test)
    acc=accuracy(kp,y_test)
    prec=precision_macro(kp,y_test)
    rec=recall_macro(kp,y_test)
    f1=macro_f1(kp,y_test)
    print("\n KNN results ")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Train Time: {knn.train_time:.4f} sec")
    results.append(("KNN", acc, prec, rec, f1, knn.train_time))

    # Final comparison table
    print("\n\n COMPARISON TABLE ")
    print("{:<15} {:>10} {:>12} {:>12} {:>10} {:>12}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1", "TrainingTime"
    ))
    for name, acc, prec, rec, f1, t in results:
        print("{:<15} {:10.2f} {:12.4f} {:12.4f} {:10.4f} {:12.3f}".format(
            name, acc, prec, rec, f1, t
        ))


if __name__=="__main__":
    run_all()
