# -*- coding:utf-8 -*-


import numpy as np

from nltk.probability import FreqDist

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn import metrics

#测试样本数


"""
def load_alexa(filename):
    domain_list=[]
    csv_reader = csv.reader(open(filename))
    for row in csv_reader:
        domain=row[1]
        if domain >= MIN_LEN:
            domain_list.append(domain)
    return domain_list
def domain2ver(domain):
    ver=[]
    for i in range(0,len(domain)):
        ver.append([ord(domain[i])])
    return ver
    #domain_list=load_alexa("../data/top-1m.csv")
    domain_list = load_alexa("../data/top-1000.csv")
    #remodel=train_hmm(domain_list)
    remodel=joblib.load(FILE_MODEL)
    x_3,y_3=test_dga(remodel, "../data/dga-post-tovar-goz-1000.txt")
    x_2,y_2=test_dga(remodel,"../data/dga-cryptolocke-1000.txt")
    x_1,y_1=test_alexa(remodel, "../data/test-top-1000.csv")
    #test_alexa(remodel, "../data/top-1000.csv")
    #%matplotlib inline
    fig,ax=plt.subplots()
    ax.set_xlabel('Domain Length')
    ax.set_ylabel('HMM Score')
    ax.scatter(x_3,y_3,color='b',label="dga_post-tovar-goz")
    ax.scatter(x_2, y_2, color='g', label="dga_cryptolock")
    #ax.scatter(x_1, y_1, color='r', label="alexa")
    ax.legend(loc='right')
    plt.show()
"""

def load_command_data(filename):
    command_list=[]
    max_freq_cmd=[]
    min_freq_cmd=[]
    all_commands=[]
    with open(filename) as f:
        i=0
        x=[]
        for line in f:
            line=line.strip('\n')
            x.append(line)
            all_commands.append(line)
            i+=1
            if i == 100:
                command_list.append(x)
                x=[]
                i=0

    command_frequence = list(FreqDist(all_commands).keys())
    max_freq_cmd=set(command_frequence[0:50])
    min_freq_cmd = set(command_frequence[-50:])
    return command_list,max_freq_cmd,min_freq_cmd

def get_commands_features(user_cmd_list,max_freq_cmd,min_freq_cmd):
    freatures=[]
    for cmd_block in user_cmd_list:
        feature1=len(set(cmd_block))
        command_frequence = list(FreqDist(cmd_block).keys())
        feature2=command_frequence[0:10]
        feature3=command_frequence[-10:]
        feature2 = len(set(feature2) & set(max_freq_cmd))
        feature3=len(set(feature3)&set(min_freq_cmd))
        x=[feature1,feature2,feature3]
        freatures.append(x)
    return freatures

def get_label(filename,index=0):
    x=[]
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            x.append( int(line.split()[index]))
    return x

if __name__ == '__main__':
    user = 3
    N_train_test_split = 100
    user_cmd_list,user_cmd_freq_max,user_cmd_freq_min=load_command_data(f"../data/masquerade-data/User{user}")
    freatures=get_commands_features(user_cmd_list,user_cmd_freq_max,user_cmd_freq_min)
    labels=get_label("../data/masquerade-data/label.txt",user-1)
    y=[0]*50+labels

    x_train=freatures[0:N_train_test_split]
    y_train=y[0:N_train_test_split]

    x_test=freatures[N_train_test_split:150]
    y_test=y[N_train_test_split:150]

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    y_predict=neigh.predict(x_test)

    score=np.mean(y_test==y_predict)*100

    #print y
    #print y_train
    print(y_test)
    print(y_predict)
    print(score)

    print(classification_report(y_test, y_predict))

    print(metrics.confusion_matrix(y_test, y_predict))
