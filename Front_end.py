
# coding: utf-8

# In[1]:


import os
from os import sys, path
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"   #0 to run on GPU, -1 to run on CPU (on PC, not on colab)
sys.path.append(r'/content/drive/MyDrive/testing')


# In[2]:

import back_end

# In[3]:

model=back_end.build_model(3,1,'adam','mse')


# In[4]:


model=back_end.train_model(2,1000,0.15,model)

# In[5]:
model=back_end.model_prediction(1,model)


