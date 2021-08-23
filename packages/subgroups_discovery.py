import os 
import numpy as np
import pandas as pd
import math
import time

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge 
import statsmodels.api as sm



def SplitBasedSelectionForm(data, k, model, list_neigh, split_point1, split_point2, nb_classes, limit)  :
    
    start_time = time.time()
    n = np.size(data,0) # number of instances
    p = np.size(data,1) # number of features

    Subgroups = set() # S is a set of the subgroups  
    Subgroups.add(tuple(np.arange(n))) # first S is simply all the objects O i.e S = {0}
    
    W = dict ()
    data_neigh_O, target_neigh_O_proba = sampling_sb(data,np.arange(n),list_neigh,model)
    patterns = dict()
    # patterns = {attribute : a, value : v, operator : '>' or '<='}
    
    patterns[tuple(np.arange(n))] = (None,None,None)
    
    L_Patterns = []
    L_S = []

    improv = True
    splits = set ()
    newSubgroups = set()
    newSubgroups.add(tuple(np.arange(n))) # newSubgroups = {O}
    
    loss_subgroups = dict () # for the losses of the subgroups without spliting
    loss_subgroups [tuple(np.arange(n))] = calc_loss(data_neigh_O,target_neigh_O_proba, limit)
    #print('loss_all = ',loss_subgroups [tuple(np.arange(n))])
    #print("%s" % (time.time() - start_time))

    iteration = 0    
    while len(Subgroups) < k and improv :
        print('trace ---', iteration)
        for s in newSubgroups : # s is tuple
            if len(s) > 1 and loss_subgroups[s] > 0 : 
                list_loss_attributes = []           
                for a in range(0,p) : 
                    to_continue = False
                    list_loss_values = []
                    min_v = np.min(data[s,a]) #  
                    max_v = np.max(data[s,a]) # 
                    if (a < split_point1) or (a >= split_point2) : # numerical / boolean
                        if min_v != max_v : 
                            steps = (pd.cut(data[s,a],2, retbins=True,include_lowest=True))[1][1:-1]
                            to_continue = True
                    
                    else : 
                        if min_v == 0 and max_v > 0 :
                            steps = np.array([0])
                            to_continue = True
                    if to_continue :    
                        
                        len_steps = np.size(steps)
                        j = 0 
                        while j < len_steps :

                            value = steps[j]

                            # subgroup1 that satisfies the condition s [a > v]
                            subgrp1 = tuple(np.asarray(s)[data[s,:][:,a] > value])

                            # generating the new dataset of neighbors of the subgroup_1 elements 
                            data_neigh_sb1, target_neigh_sb1_proba = sampling_sb(data,subgrp1,list_neigh,model)

                            # subgroup2 that satisfies the condition s [a <= v]
                            subgrp2 = tuple(np.asarray(s)[data[s,:][:,a] <= value])

                            # generating the new dataset of neighbors of the subgroup_1 elements 
                            data_neigh_sb2, target_neigh_sb2_proba = sampling_sb(data,subgrp2,list_neigh,model)

                            #compute the loss and update the loss_subgroups dictionnary

                            loss_subgroups[subgrp1] = calc_loss(data_neigh_sb1, target_neigh_sb1_proba, limit)
                            loss_subgroups[subgrp2] = calc_loss(data_neigh_sb2, target_neigh_sb2_proba, limit)

                            loss =  loss_subgroups[subgrp1] + loss_subgroups[subgrp2]
                            #print("loss des 2 sbgrps =",loss)
                            
                            
                            # store the losses 
                            list_loss_values.append((loss,value))

                            #iterate over the j
                            j += 1 

                    # select the minimum loss and value that minimize the loss for each attribute a 
                    if list_loss_values :

                        loss_opt_att = min(list_loss_values)

                        # store the optimal loss for the attribute 
                        list_loss_attributes.append(loss_opt_att)
                    
                    else :
                        list_loss_attributes.append((math.inf,None))

                # select the minimum loss and value that minimize the loss for the subgroup s
                loss_opt_s, value_opt = min(list_loss_attributes)
                attribute_opt = list_loss_attributes.index(min(list_loss_attributes))

                # add the best split for the subgroup (s) to the splits set 
                splits.add((s,attribute_opt,value_opt,loss_opt_s))
        
        # Choose the subgroup split that leads to the minimum loss:
        best_split =  splits.pop()
        tmp_split = best_split # to add it after 
        
        s, a, v, loss_sb = best_split
        
        Subgroups.remove(s)
        best_loss_s = loss_set(Subgroups,loss_subgroups) + loss_sb
        Subgroups.add(s)
        
        for split in splits :
            s_, a_, v_, loss_sb_ = split
            Subgroups.remove(s_)
            if loss_set(Subgroups,loss_subgroups) + loss_sb_ < best_loss_s :
                best_loss_s = loss_set(Subgroups,loss_subgroups) + loss_sb_
                best_split = split
            Subgroups.add(s_)
            
        splits.add(tmp_split)
        
        s_best, a_best, v_best, loss_sb_min = best_split
        
        if loss_sb_min < loss_subgroups[s_best] :
                        
            Subgroups.remove(s_best)
            
            sb1 = tuple(np.asarray(s_best)[data[s_best,:][:,a_best] > v_best])           
            sb2 = tuple(np.asarray(s_best)[data[s_best,:][:,a_best] <= v_best])

            Subgroups.add(sb1)
            Subgroups.add(sb2)
            
            if iteration == 0 :
                del patterns[s_best]
                patterns[sb1] = (a_best,'>',v_best)
                patterns[sb2] = (a_best,'<=',v_best)
                
            else : 
                
                patterns[sb1] = patterns[s_best] + (a_best,'>',v_best)
                patterns[sb2] = patterns[s_best] + (a_best,'<=',v_best)
                del patterns[s_best]
            
            newSubgroups = {sb1, sb2}
            splits.remove(best_split)
        else :
            improv = False
        
        iteration = iteration + 1
        #print('{:.2e}'.format(loss_set(Subgroups,loss_subgroups)))
        #print("%s" % (time.time() - start_time))

        S_copy = set ()
        S_copy = Subgroups.copy()
        
        patterns_copie = dict ()
        patterns_copie = patterns.copy()
        
        L_Patterns.append(patterns_copie)
        L_S.append(S_copy)
        
    return(L_S, L_Patterns)


def lin_models_for_sim(S,data_test,list_neigh,model,limit) :
    
    W_ = dict()
    for s in S :

        data_neigh_s, target_neigh_s_proba = sampling_sb(data_test,s,list_neigh,model)
        
        lr = Ridge(alpha = 1)
        model_lr = lr.fit(data_neigh_s[:,:limit],target_neigh_s_proba)
        W_[s] = model_lr
        del lr
        del model_lr           
    return W_

def sampling_sb(dataset, subgroup, list_neigh, model) :
    
    n_neighs = list_neigh[0][0].shape[0]
    subgroup = np.asarray(subgroup)
    size = subgroup.size
    all_data = np.zeros((size*n_neighs, dataset.shape[1]))
    all_target = np.zeros((size*n_neighs, 19))
      
    for i in range(0,subgroup.size) :
        all_data[i*n_neighs : (i+1)*n_neighs,:] = list_neigh[subgroup[i]][0]
        all_target[i*n_neighs : (i+1)*n_neighs,:] = list_neigh[subgroup[i]][1]
            
    return (all_data, all_target)


def calc_loss (data,target_proba, limit) :
    
    lr = Ridge(alpha = 1)
    model_lr = lr.fit(data[:,:limit],target_proba)
    target_lr = model_lr.predict(data[:,:limit])
    return sum(sum(np.square(target_proba-target_lr)))

def loss_set(Subgroups, loss_subgroups):
    
    loss = 0
    if bool (Subgroups) == False  : #empty
        return 0
    
    else : 
        for s in Subgroups :
            loss = loss + loss_subgroups[s]

        return loss
