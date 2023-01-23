#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTS
from sklearn import datasets
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from minisom import MiniSom
from hyperopt import Trials , STATUS_OK ,fmin, tpe, hp
from statistics import mode


# In[10]:


class solver:
    def __init__(self, som_grid_rows, som_grid_columns, sigma, iterations, learning_rate):
        self.som_grid_rows = som_grid_rows
        self.som_grid_columns = som_grid_columns
        self.sigma = sigma
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    '''to find the distance between two points in space'''
    def dist(self, a, b):
        ans=0
        for i in range(len(a)):
            ans+= (a[i]-b[i])**2 
        return ans**0.5
    
    def score(self, node, label, dict_, label_cnt):
        arr = [[self.dist(i, node), mode(dict_[i])] for i in dict_]
        arr.sort(key=lambda x :x[0])
        constant = 0.001
        if label>=len(label_cnt):
            label = len(label_cnt)-1
        val = arr[:label_cnt[label]]
        sumi = 0
        for i in val:
            if i[1]==label:
                sumi+=1
        return (sumi/label_cnt[label])+constant
    
    def get_label_count(self, dict_):
        label_cnt = {}
        for i in dict_:
            mode_ =  mode(dict_[i])
            if mode_ in label_cnt:
                label_cnt[mode_]+=1
            else:
                label_cnt[mode_]=1
        return label_cnt
    
    '''this tunes the weights of our som, 
        returning the best sigma and learning rates received after quantization'''
    def tuning(self, df_75):
        space = {
            "sig": hp.uniform("sig",0.001,5),
            "learning_rate" : hp.uniform("learning_rate",0.001,5)
        }
        def som_fn(space):
            sig = space["sig"]
            learning_rate = space["learning_rate"]
            val = MiniSom (x=self.som_grid_rows ,
                           y=self.som_grid_columns, 
                           input_len = df_75.shape[1],
                           sigma =self.sigma,
                           learning_rate = self.learning_rate
                          ).quantization_error(df_75)
            print(val)
            return {'loss':val , 'status':STATUS_OK}
        som_fn(space)
        trials = Trials()
        best = fmin(
                fn = som_fn,
                space= space,
                algo = tpe.suggest, 
                max_evals = 1000,
                trials =trials
        )
        sigma = best["sig"]
        learning_rate = best["learning_rate"]
        return sigma, learning_rate
#         print(f"x :{x}\ny : {y}\nsigma :{sigma}\nlearning_rate :{learning_rate}")
    
    '''this function randomly intitializes the weights of som, 
        based on the characteristics provided by the user'''
    def train_som(self, df_75):
        som = MiniSom(x = som_grid_rows, 
              y = som_grid_columns,
              input_len = df_75.shape[1],
              sigma = self.sigma, 
              learning_rate = self.learning_rate)
        som.random_weights_init(df_75)

        som.train_random(df_75, self.iterations)
        return som
      
    '''this loads the dataset from the csv file and returns the numpy array after sampling the 25% of the data'''
    def load_dataset(self, path):
        df = pd.read_csv(path)
        df=df.iloc[:,1:]
        # 0: prevotella, 1: bacteroids, 2: ruminoccocus
        arr = np.array(df.iloc[:,[0,1,5]])
        
        labels = [np.argmax(i) for i in arr]
        df["class"]=labels
        
        df_75 = df.sample(frac = 0.75)
        df_25 = df.drop(df_75.index)
        labels_75= np.array(df_75.iloc[:,-1])
        labels_25= np.array(df_25.iloc[:,-1])
        
        df_75 = df_75.iloc[:,:-1]
        df_25 = df_25.iloc[:,:-1]
        return df, np.array(df_75), np.array(df_25), labels_75
    
    '''this function calculates the representative node from each of the clusters sample data points'''
    def represen_node_label(self, df, label, som):
        # here data is the dataframe
        df_ = np.array(df[df["class"]==label].iloc[:,:-1])
        final_dict, dict = {}, {}

        for cnt, xx in enumerate(df_):
            w = som.winner(xx)
            key = (w[0], w[1])
            if key in dict.keys():
                dict[(w[0], w[1])]+=1
            else:
                dict[(w[0], w[1])]=1
        final_x, final_y, vals = 0, 0, 0

        for key, value in dict.items():
            x, y = key[0]*value, key[1]*value
            final_x += x
            final_y += y
            vals +=value
        final_x = final_x/vals
        final_y = final_y/vals
        final_dict[label] = (final_x, final_y)
        mean=final_dict[label]
        ans=float("inf")
        res=(0,0)
        for j in dict:
            if self.dist(j, mean)<ans:
                ans = self.dist(j, mean)
                res= j
        return res
    
    '''this function plots the nodes on the kohonen map, representing the wide distribution of nodes 
        into the respective clusters'''
    def plot_som(self, som, df_75, target):
        plt.figure(figsize=(9,9))
        plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
        plt.colorbar()
        markers = ['o', 's', 'D']
        colors = ['r', 'g', 'b']
        for cnt, xx in enumerate(df_75):
            w = som.winner(xx)  # getting the winner
            # palce a marker on the winning position for the sample xx
            plt.plot(w[0]+.5, w[1]+.5, markers[target[cnt]], markerfacecolor='None',
                     markeredgecolor=colors[target[cnt]], markersize=12, markeredgewidth=2)

        plt.show()
        
    '''this function gives the node with the cluster of the respective sample which hits that node'''
    def getting_labels(self, df, som):
        dict = {}
        for label in range(3):
            df_ = np.array(df[df["class"]==label].iloc[:,:-1])

            for cnt, xx in enumerate(df_):
                w = som.winner(xx)
                key = (w[0], w[1])
                if key in dict.keys():
                    dict[(w[0], w[1])].append(label)
                else:
                    dict[(w[0], w[1])] = [label]
        return dict

    '''this function gives the confidence score for that respective sample, 
         to belong to the respective cluster (it ranges b/w 0 to 1)'''
    def getting_confidence_score(self, representative_nodes, dict_, label_cnt, df):
        node_cnf={}
        conf_0, conf_1, conf_2=[], [], []
        test_df ={"conf_0":conf_0 ,"conf_1":conf_1,"conf_2":conf_2}#,"actual_label":df['class']}
        node_conf={}
        e = np.e
        emax = 708
        
        for i in range(0,7):
            for j in range(0,7):
                node = (i,j)

                c2=np.array([self.dist(node,representative_nodes[k])+0.001 for k in range(3)])
                c1=np.array([self.score(node,k,dict_,label_cnt) for k in range(3)])

                print(node, c1, c2)
                final_conf = c1/c2
                if(final_conf[0]>emax):
                    final_conf[0] = 708
                if(final_conf[1]>emax):
                    final_conf[1] = 708
                if(final_conf[2]>emax):
                    final_conf[2] = 708
                final_ans=[
                    (e**final_conf[0])/(e**final_conf[0] + e**final_conf[1] + e**final_conf[2]),
                    (e**final_conf[1])/(e**final_conf[0] + e**final_conf[1] + e**final_conf[2]),
                    (e**final_conf[2])/(e**final_conf[0] + e**final_conf[1] + e**final_conf[2])]
                node_cnf[node]=final_ans

        dff = np.array(df.iloc[:, :-1])
        for i in dff:
            win_node = som.winner(i)
            conf= node_cnf[win_node]
            conf_0.append(round(conf[0], 3))
            conf_1.append(round(conf[1], 3))
            conf_2.append(round(conf[2], 3))
        
        self.test_df=pd.DataFrame(test_df)
        
        return self.test_df
    
    def most_representative(self, df_25, cluster, top_n):
        ind=[]
        for i in range(len(self.test_df)):
            if test_df.iloc[i,cluster]==max(self.test_df.iloc[i,0], self.test_df.iloc[i,1], self.test_df.iloc[i,2]):
                ind.append(i)
        df_ = df_25[ind,:]
        d={}
        for i in range(df_25.shape[1]):
            temp = df_[:,i]
            d[i]=np.var(temp)
#         print(d)
        sorted_d = sorted(d.items(), key=lambda x:x[1])
        sorted_d = dict(sorted_d)
#         print(sorted_d)
        return list(d.keys())[::-1][:top_n]

    def plot_winner(self, a, som, df_75, target):
        plt.figure(figsize=(9,9))
    #     bone()
        plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
        plt.colorbar()
        markers = ['o', 's', 'D']
        colors = ['r', 'g', 'b']
        for cnt, xx in enumerate(df_75):
            w = som.winner(xx)  # getting the winner
            # palce a marker on the winning position for the sample xx
            plt.plot(w[0]+.5, w[1]+.5, markers[target[cnt]], markerfacecolor='None',
                     markeredgecolor=colors[target[cnt]], markersize=12, markeredgewidth=2)

        w=som.winner(a)
        plt.plot(w[0]+0.5,w[1]+0.5,"X",markerfacecolor='YELLOW',markeredgecolor="YELLOW", markersize=50, markeredgewidth=2)
        plt.show()
        
    def marginal(self, marginal, data, neurons):
        # check if the second argument is of type character
        if type(marginal) == str and marginal in list(data):

            f_ind = list(data).index(marginal)
            f_name = marginal
            train = np.matrix(data)[:, f_ind]
            neurons = neurons[:,:, f_ind].flatten()
            print(train.shape)
            print(neurons.shape)
            plt.ylabel('Density')
            plt.xlabel(f_name)
            sns.kdeplot(np.ravel(train),
                   label="training data",
                            shade=True,
                            color="b")
            #print(neurons)
            sns.kdeplot(neurons, label="neurons", shade=True, color="r")
            plt.legend(fontsize=15)
            plt.show()


# In[11]:


som_grid_rows = 7
som_grid_columns = 7
sigma = 1
iterations = 1000
learning_rate = 0.5


# In[12]:


s = solver(som_grid_rows, som_grid_columns, sigma, iterations, learning_rate)
path = '../METAHIT.csv'
df, df_75, df_25, target = s.load_dataset(path)
df.head()


# In[13]:


# train som model (som_grid_rows, som_grid_columns, sigma, iterations, learning_rate)
som = s.train_som(df_75)
s.plot_som(som, df_75, target)     #  (som, df_75, target)


# In[14]:


# tuning (x, y, input_len)
sigma, learning_rate = s.tuning(df_75)


# In[15]:


som = s.train_som(df_75)
s.plot_som(som, df_75, target)


# In[16]:


# getting representative nodes
representative_nodes = [s.represen_node_label(df, i, som) for i in range(0,3)]


# In[ ]:





# In[17]:


# getting labels
dict_ = s.getting_labels(df, som)
label_cnt = s.get_label_count(dict_)


# In[18]:


print(label_cnt)


# In[19]:


# print confidence score (representative_nodes, dict_, label_cnt)
conf_scores = s.getting_confidence_score(representative_nodes, dict_,label_cnt,df)
print(conf_scores)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




