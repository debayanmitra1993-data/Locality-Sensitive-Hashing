import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class LSH:

  def __init__(self,num_hyperplanes = 8,max_iter = 100):
    self.num_hyperplanes = num_hyperplanes
    self.w1_list = []
    self.w2_list = []
    self.hash_maps = {}
  
  def fit(self,X):
    self.X_train = X

    # generate hyperplanes                    
    for i in range(self.num_hyperplanes):
      w1 = np.random.normal()
      w2 = np.random.normal()
      self.w1_list.append(w1)
      self.w2_list.append(w2)
    
    # assign each point to signed hash bucket
    for i in range(self.X_train.shape[0]):
      hash_list = []
      for j in range(self.num_hyperplanes):
        if (self.w1_list[j]*self.X_train[i][0]) + (self.w2_list[j]*self.X_train[i][1]) >= 0:
          hash_list.append(1)
        else:
          hash_list.append(-1)
      if tuple(hash_list) in self.hash_maps:
        self.hash_maps[tuple(hash_list)].append(i)
      else:
        self.hash_maps[tuple(hash_list)] = [i]
  
  def fx(self,x,a,b):
    return -(a/b)*x
    
  def plot_partition_spaces(self):
    colors = matplotlib.cm.get_cmap("Accent").colors
    plt.figure(figsize = (12,12))
    plt.scatter(self.X_train[:,0], self.X_train[:,1], c='red')
    for i in range(self.num_hyperplanes):
      plt.plot(np.array([np.min(self.X_train[:,0]),np.max(self.X_train[:,0])]), 
               self.fx(np.array([np.min(self.X_train[:,0]),np.max(self.X_train[:,0])]), self.w1_list[i], self.w2_list[i]), 
               c = colors[i % len(colors)])
      plt.xlim([np.min(self.X_train[:,0]),np.max(self.X_train[:,0])])
      plt.ylim([np.min(self.X_train[:,1]),np.max(self.X_train[:,1])])
    plt.show()
  
  # predict for one query point
  def predict(self,x):
    hash_list = []
    for j in range(self.num_hyperplanes):
      if (self.w1_list[j]*x[0]) + (self.w2_list[j]*x[1]) >= 0:
        hash_list.append(1)
      else:
        hash_list.append(-1)
    return self.hash_maps[tuple(hash_list)]
