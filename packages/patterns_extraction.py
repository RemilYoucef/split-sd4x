import os 
import numpy as np
import pandas as pd

from neighbors_generation import *

def patterns(P, split_point1, split_point2, data, att_names_):
    
    
    patt_dict = {}
    for rank, (s, p) in enumerate(P.items()):
        
        d = {}
        for it in range(0, len(p), 3):
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

        print ('subrgoup',rank)

        description = ''
        for att, value in d.items():
            if (
                att >= split_point1
                and att < split_point2
                and value[1] == 0
                or att >= split_point1
                and att >= split_point2
                and value[0] < 0.5
            ):
                print(att_names_[att],"=",'0')
                description += att_names_[att] + ' = ' + '0' +'  \n'
            elif (
                att >= split_point1
                and att < split_point2
                or att >= split_point1
            ):
                print(att_names_[att],"=",'1')
                description += att_names_[att] + ' = ' + '1' +'  \n'

            else: 
                print(round(value[0]*23,0),"<",att_names_[att],"<=",round(value[1]*23,0))
                description += (
                    f'{str(round(value[0]*23,0))} < '
                    + att_names_[att]
                    + ' <= '
                    + str(round(value[1] * 23, 0))
                    + '  \n'
                )


        patt_dict[s] = description
        print("-------------------------------------------------------------------")
    return patt_dict

