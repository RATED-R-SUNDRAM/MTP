#!/usr/bin/env python
# coding: utf-8

#IMPORTED REQUIREED LIBRARIES AND PACKAGES

from sklearn import datasets
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics.pairwise import euclidean_distances
from minisom import MiniSom
from hyperopt import Trials , STATUS_OK ,fmin, tpe, hp
from statistics import mode


# MAIN CLASS FUNCTION NAMED SOM_EN

class SOM_EN:
    
    
    def __init__(self, som_grid_rows, som_grid_columns, sigma, iterations, learning_rate):
        """ INITIALIZATION OF SOM PARAMAETERS 
         * som_grid_rows , som_grid_columns -> the map dimensions
         * sigma -> the initial spread of the gaussian neighbourhood function
         * learning_rate -> factor with which the weight updation is done
         * iteration -> The number of times the entire dataset is passed through the training process
         """ 
        
        self.som_grid_rows = som_grid_rows
        self.som_grid_columns = som_grid_columns
        self.sigma = sigma
        self.iterations = iterations
        self.learning_rate = learning_rate
        
    
    '''Helper functions to be used in other main fuctions'''
    
    def dist(self, a, b):
        """ Func for euclidean distance of 2 arrays a and b""" 
        
        ans=0
        for i in range(len(a)):
            ans+= (a[i]-b[i])**2 
        return ans**0.5
    
    def score(self, node, label, dict_, label_cnt): 
        
        """ This functon is used to to provide score to each node in
            the map by taking in the below paramaters 
            PARAMS: 
            
            * node -> A particular node's cordinate (x,y)
            * label -> confidence score for a particular class
            * dict_ -> It is a mapping for each node which class training datasets have found hit
            * label_cnt -> Count of nodes hit by each class during training
            """ 
        
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
        
        """ Function to calculate the label_cnt dictionary in score function
            PARAMS :
             * dict_ ->It is a mapping for each node which class training datasets have found hit
         """
        
        label_cnt = {}
        for i in dict_:
            mode_ =  mode(dict_[i])
            if mode_ in label_cnt:
                label_cnt[mode_]+=1
            else:
                label_cnt[mode_]=1
        return label_cnt
    
    
    def tuning(self, df_75):
        
        
        '''This function tunes the Hyperparameters of our som, 
        returning the best sigma and learning rates received after optimization
          PARAMS : 
          * df_75 -> train dataset on which the optimization on search space takes place
        '''
        
        
        space = {
            
            # the space in which we are looking to acheive the optimal values of hyperparameters
            
            "sig": hp.uniform("sig",0.001,5),
            "learning_rate" : hp.uniform("learning_rate",0.001,5)
        }
        
        def som_fn(space):
            sig = space["sig"]
            learning_rate = space["learning_rate"]
            val = MiniSom (
                
                           # As entered by user while creating the object
                
                           x=self.som_grid_rows ,
                           y=self.som_grid_columns, 
                           input_len = df_75.shape[1],
                           sigma =self.sigma,
                           learning_rate = self.learning_rate
                          ).quantization_error(df_75)          # the optimization is on a loss/error function
                                                               # In this case it is quantization error
            #print(val)
            return {'loss':val , 'status':STATUS_OK}
        
        # optimizing in the search space and on the loss function defined
        som_fn(space)
        trials = Trials()
        best = fmin(
                fn = som_fn,
                space= space,
                algo = tpe.suggest, 
                max_evals = 1000,
                trials =trials
        )
        
        sigma = best["sig"]   # optimized / tuned hyperparameter sigma 
        learning_rate = best["learning_rate"]  # optimized /tuned hyperparameter learning_rate
        return sigma, learning_rate
#       print(f"x :{x}\ny : {y}\nsigma :{sigma}\nlearning_rate :{learning_rate}")
    
    

    def train_som(self, df_75):
        
        """ This function calls the training of som by randomly intitializing the weights of som, 
        based on the parameters provided by the user .
        PARAMS :
        df_75 -> dataset upon which the training is done.
        """
        
        som = MiniSom(
              
              # As entered by user while creating the object
            
              x = som_grid_rows,    
              y = som_grid_columns,         
              input_len = df_75.shape[1],
              sigma = self.sigma, 
              learning_rate = self.learning_rate)
        
        som.random_weights_init(df_75)   # random weight initialization 

        som.train_random(df_75, self.iterations) 
        return som
      
   
    def load_dataset(self, path):
        
        """ The whole pipeline of this class uses numpy arrays inplace of dataframe so this
            function after taking input from user in form of dataframe converts to numpy
            array and also does train - test split
            PARAMS : 
            path -> path of the csv file
            
            
        """
            
        df = pd.read_csv(path)
        df=df.iloc[:,1:]
        # 0: prevotella, 1: bacteroids, 2: ruminoccocus in case of Enterotypes
        arr = np.array(df.iloc[:,[0,1,5]])
        
        labels = [np.argmax(i) for i in arr]  # taking the maximum as a class label for enterotypes otherwise provided by user
        df["class"]=labels
        
        df_75 = df.sample(frac = 0.75)
        df_25 = df.drop(df_75.index)
        labels_75= np.array(df_75.iloc[:,-1])
        labels_25= np.array(df_25.iloc[:,-1])
        
        df_75 = df_75.iloc[:,:-1]
        df_25 = df_25.iloc[:,:-1]
        return df, np.array(df_75), np.array(df_25), labels_75
    
  
    def represen_node_label(self, df, label, som):
        
        
        """This function calculates the representative node from each of the clusters sample data points basically 
           the mathematically mean or centre of each node for a particular class
           PARAMS :
           df -> entire dataset passed
           label -> class or cluster which representative node is to be calculated
           som -> the som object in which weight initialization and paramters are trained
           
        """
        
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
    
    
    def plot_som(self, som, df_75, target):
        
        '''This function plots the nodes on the kohonen map, representing the wide distribution of nodes 
        into the respective clusters
        PARAMS : 
        som -> som object in which weight initialization and paramters are trained
        df_75 -> train dataset
        target -> class array
        
        '''
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
        
    
    def getting_labels(self, df, som):
        
        """This function gives the mapping of each node with the cluster of the respective sample
           which hits that node
           PARAMS :
           df -> the entire dataset
           som -> som object in which weight initialization and paramters are trained
           
        """
        
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

    
    def getting_confidence_score(self, representative_nodes, dict_, label_cnt, df):
        
        '''This function gives the confidence score for that respective sample, 
         to belong to the respective cluster (it ranges b/w 0 to 1)
         PARAMS :
         representative_nodes -> calculated from the function represen_node_label
         dict_ -> It is a mapping for each node which class training datasets have found hit
         label_cnt -> Calculated from the function get_label_count
         df -> entered dataframe 
         '''
        
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
        
        """This function calculates the most important or contributing 
           features in creating seperable clusters 
           PARAMS : 
           df_25 -> test dataset
           cluster -> class in which contributing features are to be calculated
           top_n -> top n most contributing features
           
        """
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

    def highlight_winner(self, a, som, df_75, target):
        
        """ Plotting the winner node in a map where each winner node is
            highlighted in respective colors
            PARAMS :
            a -> array which winner node is to be highlited
            som -> som object in which weight initialization and paramters are trained
            df_75 -> train dataset
            target -> class array
            
        """ 
        
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
        
        """ This function plots the comparitve marginal density function of 
            both map weights after training and data to see how well our 
            map has covered the dataset.
            PARAMS: 
            marginal : the feature whose plot is to be plotted. 
            data -> dataset entered 
            neurons -> weights of the som object """ 
        
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






