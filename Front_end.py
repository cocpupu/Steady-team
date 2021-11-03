
# coding: utf-8

# In[1]:


import os
from os import sys, path
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
sys.path.append(r'/content/drive/MyDrive/DFM_FR_2020/Deep-Learning-for-Aerodynamic-Prediction-master/test_1_img')


# In[2]:

import back_end

# In[3]:

model=back_end.build_model(3,1,'adam','mse')

# In[4]:

0
model=back_end.train_model(5,100,0.15,model)

# In[5]:
model=back_end.model_prediction(1,model)


