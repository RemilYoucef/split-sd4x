import os 
import numpy as np
import pandas as pd

from neighbors_generation import *

def patterns (P, split_point1, split_point2, data, att_names_) :
    
    
    patt_dict = dict()
    rank = 0
    for s,p in P.items() :
        
        description = ''
        it = 0
        d = dict ()
        while (it < len(p)) :
            a,op,v = p[it],p[it+1],p[it+2]
            if a not in d :
                d[a] = [np.min(data[:,a]) ,
                        np.max(data[:,a]) ]

            if op == '>' :
                #update le min
                d[a][0] = max(v,d[a][0])

            else : 
                #update le max
                d[a][1] = min(v,d[a][1])

            it += 3
                                                
        print ('subrgoup',rank)
        
        for att, value in d.items():
            if att < split_point1 : 
                print(round(value[0]*23,0),"<",att_names_[att],"<=",round(value[1]*23,0))
                description += str(round(value[0]*23,0)) + ' < ' + att_names_[att] + ' <= ' + str(round(value[1]*23,0)) +'  \n'
            
            elif att < split_point2 :
                if value[1] == 0 :
                    print(att_names_[att],"=",'0')
                    description += att_names_[att] + ' = ' + '0' +'  \n'
                else :
                    print(att_names_[att],"=",'1')
                    description += att_names_[att] + ' = ' + '1' +'  \n'
                
            else :
                if value [0] < 0.5  : 
                    print(att_names_[att],"=",'0')
                    description += att_names_[att] + ' = ' + '0' +'  \n'
                    
                else :
                    print(att_names_[att],"=",'1')
                    description += att_names_[att] + ' = ' + '1' +'  \n'
         
        patt_dict[s] = description 
        print("-------------------------------------------------------------------")
        rank += 1
    
    return patt_dict

