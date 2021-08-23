
import os.path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def plot_explanations(W, S_list, nb_subgroups, c, att_names, patt_descriptions) : 
    
    for j in range(0, nb_subgroups) :
        
        print(j,'------------------------------------------------')
        print(patt_descriptions[S_list[j]]) 
        coefficients = W[S_list[j]].coef_
        logic = coefficients > 0
        coefficients_abs = np.abs(coefficients)
        contributions = coefficients_abs / np.sum(coefficients_abs, axis = 1).reshape(-1,1)
        features_importance = contributions[c] * 100
        limit = 0.75
        
        f_importance = features_importance[features_importance > limit]
        f_importance = f_importance / np.sum(f_importance) * 100
        f_importance = f_importance.round(2)
        att_names_ = list(pd.Series(att_names[:362])[features_importance > limit])

        
        f_importance_1 = f_importance[logic[c][features_importance > limit]]
        att_names_1 = [x for i,x in enumerate (att_names_) if logic[c][features_importance > limit][i]]
        
        f_importance_2 = f_importance[~logic[c][features_importance > limit]]
        att_names_2 = [x for i,x in enumerate (att_names_) if not logic[c][features_importance > limit][i]]
        
        plt.style.use('fivethirtyeight')
        plt.figure(figsize =(3, 4))
        plt.barh(att_names_2, f_importance_2,color='#e74c3c',height=0.65)
        plt.barh(att_names_1, f_importance_1,color='#1abc9c',height=0.65)        
        all_f_importance = np.concatenate((f_importance_2,f_importance_1))
        for i, v in enumerate(all_f_importance) :
            plt.text(v + 0.4, i, str(v)+'%', fontsize = 9)
        
        plt.xlabel("Features Importance",fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', which='major',color='grey', alpha=0.75)
        plt.savefig('FIGURES/f_'+str(j))
        plt.show()

def sort_subgroups_support(S,K) :
	S_copy = S.copy()
	l_best_s = []
	for i in range(0,K) :
		inter = 0
		s_best = None 

		for s in S_copy :
			if len(s) > inter :
				inter = len(s)
				s_best = s
		l_best_s.append(s_best)
		S_copy.remove(s_best)
	
	return l_best_s