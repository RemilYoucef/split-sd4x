import os 
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from subgroups_discovery import *
from neighbors_generation import *
import matplotlib.pyplot as plt

def loss_sd (S,data_test,list_neigh,model, limit) :

    loss = 0 
    for s in S :
        data_neigh_s, target_neigh_s_proba = sampling_sb(data_test,s,list_neigh,model)
        loss += calc_loss(data_neigh_s, target_neigh_s_proba, limit)
    return loss

def loss_global_wb (data_test,list_neigh,model, limit) :

    n = np.size(data_test,0)
    data_neigh_O, target_neigh_O_proba = sampling_sb(data_test,np.arange(n),list_neigh,model)
    global_loss = calc_loss(data_neigh_O, target_neigh_O_proba, limit)
    return global_loss


def loss_local_models (n,list_neigh,model, limit) :

    loss = 0
    for i in range(0,n) :
        data_neigh_i= list_neigh[i][0]
        target_neigh_i_proba = list_neigh[i][1]
        loss += calc_loss(data_neigh_i, target_neigh_i_proba, limit)    
    return loss

def fscore_global_wb (data_test,n,list_neigh,model,nb_classes) :
    
    data_neigh_O, target_neigh_O_proba = sampling_sb(data_test,np.arange(n),list_neigh,model)
    lr = Ridge(alpha = 1)
    model_lr = lr.fit(data_neigh_O,target_neigh_O_proba)
    target_lr = model_lr.predict(data_neigh_O)
    a = np.argsort(target_lr, axis=1)[:,-3:]
    b = np.argsort(target_neigh_O_proba, axis=1)[:,-3:]

    return (f1_score(a[:,2],b[:,2],average='weighted'), f1_score(a[:,1],b[:,1],average='weighted'), f1_score(a[:,0],b[:,0],average='weighted'))

def fscore_sd (S,data_test,list_neigh,model,nb_classes) :

    iteration = 0 
    for s in S :
        data_neigh_s, target_neigh_s_proba = sampling_sb(data_test,s,list_neigh,model)
        lr = Ridge(alpha = 1)
        model_lr = lr.fit(data_neigh_s,target_neigh_s_proba)
        target_lr = model_lr.predict(data_neigh_s)
        if iteration == 0 :
            a = np.argsort(target_lr, axis=1)[:,-3:]
            b = np.argsort(target_neigh_s_proba, axis=1)[:,-3:]

        else :
            a = np.concatenate((a,np.argsort(target_lr, axis=1)[:,-3:]))
            b = np.concatenate((b,np.argsort(target_neigh_s_proba, axis=1)[:,-3:]))

        iteration += 1
        
    return (f1_score(a[:,2],b[:,2],average='weighted'), f1_score(a[:,1],b[:,1],average='weighted'), f1_score(a[:,0],b[:,0],average='weighted'))

def fscore_local_models (data_test,n,list_neigh,model,nb_classes) :
    
    
    iteration = 0 
    for i in range(0,n) :
        
        data_neigh_i= list_neigh[i][0]
        target_neigh_i_proba = list_neigh[i][1]
        lr = Ridge(alpha = 1)
        model_lr = lr.fit(data_neigh_i,target_neigh_i_proba)
        target_lr = model_lr.predict(data_neigh_i)
        if iteration == 0 :
            a = np.argsort(target_lr, axis=1)[:,-3:]
            b = np.argsort(target_neigh_i_proba, axis=1)[:,-3:]
        else :
            a = np.concatenate((a,np.argsort(target_lr, axis=1)[:,-3:]))
            b = np.concatenate((b,np.argsort(target_neigh_i_proba, axis=1)[:,-3:]))
        
        iteration += 1    
    
    return (f1_score(a[:,2],b[:,2],average='weighted'), f1_score(a[:,1],b[:,1],average='weighted'), f1_score(a[:,0],b[:,0],average='weighted'))

def unit_vector(vector):
    
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def similarity (W,nb_classes) :

    l = []

    for key, value in W.items():
        temp = [key,value]
        l.append(temp)


    distance_matrix = np.zeros(len(l)**2).reshape(len(l),len(l))
    for i in range (0,len(l)) :
        for j in range (i,len(l)):
            for c in range (0,nb_classes) :
                if c == 0 : 
                    v1 = l[i][1].coef_[c]
                    v2 = l[j][1].coef_[c]
                else :
                    v1 = np.concatenate((v1,l[i][1].coef_[c]),axis=0)
                    v2 = np.concatenate((v2,l[j][1].coef_[c]),axis=0)                    
            distance_matrix[i,j] = round(math.cos(angle_between(v1,v2)),2)
            distance_matrix[j,i] = distance_matrix[i,j]

    return distance_matrix


def avg_non_similar (dist,treshold) :
    
    nb_non_sim = 0 
    nb_sbgrps = np.size(dist,0)
    for i in range (0, nb_sbgrps) :
        for j in range (i+1, nb_sbgrps) :
            if dist[i,j] <= treshold :
                nb_non_sim += 1

    return nb_non_sim / (nb_sbgrps * (nb_sbgrps - 1) / 2)



def plot_box_plots(nb_models,list_subgroups) :
    
    L_bp = []

    for i,j in enumerate(nb_models) :
        L_bp.append([])
        S = list_subgroups[j-2]
        for s in S :
            L_bp[i].append(len(s))

    L_bp.reverse()
    fig = plt.figure(figsize =(4, 1))
    box=plt.boxplot(L_bp,vert=0,patch_artist=True,labels=['25 sg'], widths = 0.4)

    colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.savefig('distribution.PNG')
    plt.show()
    return L_bp
    

def extract_stats(l,nb_objs) :
    print('la moyenne est',int(nb_objs/len(l)))
    tmp = np.percentile(l, [25, 50, 75])
    
    print('la mediane est :',int(tmp[1]))
    print('le quartile 1 est :',int(tmp[0]))
    print('le quartile 3 est :',int(tmp[2]))
    print('la plus grande valeur est :',max(l))
    print('la plus petite valeur est :',min(l))







