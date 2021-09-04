import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import streamlit as st



    
def datafix(data,label,factor,proportion=0.4):  #数据整理
    X,y= data[factor].values,data[label].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=proportion)
    return X_train, X_test, y_train, y_test

def Average_accuracy(ML,data,label,factor,proportion=0.4,time=10): #准确率平均计算
    pre = []
    for i in range(time):
        pre.append(ML(data,label,factor,proportion=proportion))

    return [max(pre),min(pre),np.mean(pre),np.std(pre)]

def OLS(data,label,factor,proportion=0.4):                     #岭回归分类器
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = RidgeClassifier().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def LR(data,label,factor,proportion=0.4):                      #logistic回归
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = LogisticRegression().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def SGD(data,label,factor,proportion=0.4):                     #随机梯度下降
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = make_pipeline(StandardScaler(),
        SGDClassifier())
    clf.fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def PT(data,label,factor,proportion=0.4):                      #Perceptron(感知器)
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = Perceptron().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def PAC(data,label,factor,proportion=0.4):                      #被动攻击算法
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = PassiveAggressiveClassifier().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def LDA(data,label,factor,proportion=0.4):                       #线性判别分析
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def QDA(data,label,factor,proportion=0.4):                       #二次判别分析
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def SVM(data,label,factor,proportion=0.4):                        #支持向量机
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = svm.SVC().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def KNC(data,label,factor,proportion=0.4):                          #最近邻
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    neigh = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    return(neigh.score(X_test, y_test))

def GPC(data,label,factor,proportion=0.4):                          #高斯过程
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    gpc = GaussianProcessClassifier().fit(X_train, y_train)
    return(gpc.score(X_test, y_test))

def CNB(data,label,factor,proportion=0.4):                          #朴素贝叶斯
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = CategoricalNB().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def DTC(data,label,factor,proportion=0.4):                           #决策树
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def RFC(data,label,factor,proportion=0.4):                           #随机森林
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = RandomForestClassifier().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def ABC(data,label,factor,proportion=0.4):                           #AdaBoost
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = AdaBoostClassifier().fit(X_train, y_train)
    return(clf.score(X_test, y_test))
 
def VC(data,label,factor,proportion=0.4):                            #投票
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = VotingClassifier().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def MPLC(data,label,factor,proportion=0.4):                             #BP神经网络(多层感知器)
    X_train, X_test, y_train, y_test = datafix(data,label,factor,proportion=proportion)
    clf = MLPClassifier().fit(X_train, y_train)
    return(clf.score(X_test, y_test))

def ML_Collective(data,label,factor,time,proportion=0.4):
    fun = ['ABC','DTC','GPC','KNC','LDA','LR','MPLC','OLS','PAC','PT','QDA','RFC','SGD','SVM','VC','CNB']
    name = ['AdaBoost','决策树','高斯过程','最近邻','线性判别','logistic回归','BP神经网络(多层感知器)','岭回归','被动攻击算法','Perceptron','二次判别','随机森林','随机梯度下降','支持向量机','投票器','朴素贝叶斯']
    result = [[],[],[],[]]
    latest_iteration = st.empty()
    my_bar = st.progress(0)
    for i in range(14):
        tag=Average_accuracy(eval(fun[i]),data,label,factor,proportion=proportion,time=time)
        result[0].append(tag[0])
        result[1].append(tag[1])
        result[2].append(tag[2])
        result[3].append(tag[3])
        latest_iteration.text(f'计算中...\n已完成: {round((i+1)/14,4)*100}%')
        my_bar.progress((i+1)/14)
    st.success('Done!')
    out = {}
    out['算法'] = name[0:14]
    out['最大值'] = result[0]
    out['最小值'] = result[1]
    out['平均值'] = result[2]
    out['标准差'] = result[3]
    return pd.DataFrame(out)