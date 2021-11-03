
# coding: utf-8

# In[1]:


#Import

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import numpy as np
#import keras
import scipy as sc
from tensorflow.keras.layers import Dense,Conv2D,Conv2DTranspose,Flatten,Input,concatenate,Reshape,Dropout,Activation,BatchNormalization,MaxPooling2D,UpSampling2D
from tensorflow.keras.callbacks import CSVLogger,TerminateOnNaN,EarlyStopping,ModelCheckpoint
import scipy.io
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import csv
from tensorflow.keras.models import load_model
from pandas import *
import skfmm
from scipy.io import loadmat
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib
#from mpl_toolkits.axes_grid1 import make_axes_locatable


class SWISH(Activation):

        def __init__(self,activation,**kwargs):
                super(SWISH,self).__init__(activation,**kwargs)
                self.__name__='swish_fn'

def swish(x):
        return (x*K.sigmoid(x))

get_custom_objects().update({'swish_fn': SWISH(swish)})

tf.keras.optimizers.Adam(learning_rate=0.001,decay=0.5)

def load_data(data_file):  
    xlist,ylist,pres,abpres,ttpres,vx,vy  = np.loadtxt(data_file,
        skiprows=1,
        unpack=True)
    
    nx = 300
    ny = 300

    xcor     = np.ones((nx,ny))
    ycor     = np.ones((nx,ny))
    P_array  = np.ones((nx,ny))
    AB_array  = np.ones((nx,ny))
    TT_array  = np.ones((nx,ny))
    U_array  = np.ones((nx,ny))
    V_array  = np.ones((nx,ny))

    for i in range(nx):
        for j in range(ny):
            if (i==0):
                xcor[i,j]     = xlist[j]
                ycor[i,j]     = ylist[j] 
                P_array[i,j]  = pres [j]/100000.
                AB_array[i,j]  = abpres [j]/100000.      #normalize 
                TT_array[i,j]  = ttpres [j]/100000.        #normalize
                U_array[i,j]  = vx   [j]/100 
                V_array[i,j]  = vy   [j]
            else:
                xcor[i,j]     = xlist[(j + nx*i)]
                ycor[i,j]     = ylist[(j + nx*i)] 
                P_array[i,j]  = pres [(j + nx*i)]/100000. 
                AB_array[i,j]  = abpres [(j + nx*i)]/100000.      #normalize 
                TT_array[i,j]  = ttpres [(j + nx*i)]/100000.        #normalize
                U_array[i,j]  = vx   [(j + nx*i)]/100 
                V_array[i,j]  = vy   [(j + nx*i)]            
        
    return xcor,ycor,P_array,AB_array,TT_array,U_array,V_array
    
#==========================================================================
#
#       Build model
#
#==========================================================================

def build_model(num_conv,num_dense,inp_optim,inp_loss):

        input1=Input(shape=(300,300,1),name='input1')
        input2=Input(shape=(2,1),name='input2')

        num_units=np.zeros(num_conv,dtype=int)
        num_kernel=np.zeros(num_conv,dtype=int)
        num_strides=np.zeros(num_conv,dtype=int)


        #------------------------------------------------------------------------------------------------------

        with open('/content/gdrive/MyDrive/Do_an/config_file.csv','r') as csvfile:
            info = csv.reader(csvfile, delimiter=',')
            info=list(info)


        user_input_channels=int(info[0][1])

        #--------------------------------------------------------------------------------------------------------

        print("Constructing the convolutional layers:")    
 #----------------------------------------------------------------------------------Convolutional Layers Construction  
        maxpling=[]
        maxpling_size=[]
        for i in range(1,num_conv+1):
            print("Convolutional Layer %d"%(i))
            inp_numfilters   = int(info[i][1])
            num_units[i-1]   = int(info[i][1])
            inp_shape        = int(info[i][2])
            num_kernel[i-1]  = int(info[i][2])
            inp_stride       = int(info[i][3])
            num_strides[i-1] = int(info[i][3])
            inp_activation   = info[i][4]
            
            if i==1:
		
                output=Conv2D(inp_numfilters,(inp_shape,inp_shape),strides=(inp_stride,inp_stride),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(input1)
                output=BatchNormalization()(output)
            else:
                output=Conv2D(inp_numfilters,(inp_shape,inp_shape),strides=(inp_stride,inp_stride),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                output=BatchNormalization()(output)
            inp_maxpling=info[i][5]
            inp_maxpling_size=int(info[i][6])
            maxpling.append(inp_maxpling)
            maxpling_size.append(inp_maxpling_size)
            if inp_maxpling=='Y':
                output=MaxPooling2D(pool_size=(inp_maxpling_size,inp_maxpling_size),strides=(inp_maxpling_size,inp_maxpling_size))(output)
                
            else:
                continue
                
 #------------------------------------------------------------------------------------       
        shape_2=output.shape
        print(shape_2)
        output=Reshape((-1,1))(output)
        output=concatenate([output,input2],axis=-2)
        output=Flatten()(output)
        shape_1=output.shape
        print(shape_1)
        
        print("Constructing the Dense Layers")
 #------------------------------------------------------------------------------------Dense Layers Construction       
        for i in range(1,num_dense+1): 
            print("Dense Layer %d"%(i))
            inp_numunits=int(info[num_conv+i][1])
            inp_activation=info[num_conv+i][2]
            
            if i==num_dense:
                output=Dense(shape_1[1]-2,activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                output=BatchNormalization()(output)
            else:
                output=Dense(inp_numunits,activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                output=BatchNormalization()(output)
        
            
 #------------------------------------------------------------------------------------
        print(output.shape)
        output=Reshape((shape_2[1],shape_2[2],shape_2[3]))(output)
        print("Creating correspondingly symmetrical Deconvolutional layers")
        
        if user_input_channels==3:
 #------------------------------------------------------------------------------------ DeConvolutional Layers Construction 
    
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output)
                if i==num_conv:
                    output=Conv2DTranspose(3,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output)
                else:
                    output=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                    output=BatchNormalization()(output)
                    
#--------------------------------------------------------------------------------------------------------------------------------  
            model=Model(inputs=[input1,input2],outputs=[output])


 #--------------------------------------------------------------------------------------------------------------------------------  
 #---------------------------------------------------------------------------------------------------------------------   
 #-----------------------------------------------------------------------------------------------------------------------------
        else:
        
            output1=output
            output2=output
            output3=output
     #-----------------------------------------------------------------------------------3-path Deconvolution output       
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output1=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output1)
                if i==num_conv:
                    output1=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output1)

                elif i==1:
                    output1=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output1)
                    output1=BatchNormalization()(output1)
                else:
                    output1=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output1)
                    output1=BatchNormalization()(output1)
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output2=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output2)
                if i==num_conv:
                    output2=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output2)
                elif i==1:
                    output2=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output2)
                    output2=BatchNormalization()(output2)
                else:
                    output2=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output2)
                    output2=BatchNormalization()(output2)
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output3=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output3)
                if i==num_conv:
                    output3=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output3)
                elif i==1:
                    output3=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output3)
                    output3=BatchNormalization()(output3)
                else:
                    output3=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output3)
                    output3=BatchNormalization()(output3)
            model=Model(inputs=[input1,input2],outputs=[output1,output2,output3])  
     #-----------------------------------------------------------------------------------------------------------------------------
     #-----------------------------------------------------------------------------------------------------------------------------

        model.compile(optimizer=inp_optim,loss='mse')

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
             json_file.write(model_json)

        model.summary()
        return model
    
#=================================================================
#
#       Train model
#
#=================================================================
#Creates a trained model using the specified algorithm.(đào tạo mô hình dựa trên các thuật tuán đã chỉ định)
def train_model(batch_sz,eps,val_splt,model):
    with open('/content/gdrive/MyDrive/Do_an/config_file_2.csv','r') as csvfile:
        info=csv.reader(csvfile,delimiter=',')
        info=list(info)#make list
        print('info',info)
    exp_no=int(info[0][0])
    user_input_channels=int(info[0][1])
    user_inp=info[0][2]
#dùng cho earlystopping and modelcheckpoint
#moni
    if user_inp=='Y':
        user_moni=info[0][3]
        inp_delta=float(info[0][4])#float biểu diễn số dưới dạng số thập phân
        inp_mineps=float(info[0][5])
    
 #===============================================================


        # reading CSV file
        data = read_csv("/content/gdrive/MyDrive/Do_an/Interpolation/label5test.csv")
        data_size=int(5)

        # converting column data to list
        #STT = data['Stt'].tolist()
        #NACA = data['Naca'].tolist()
        Re = data['Reynold'].tolist()
        A = data['AoA'].tolist()

        x_train_2 = np.ones((data_size,2))#Tạo một numpy array với tất cả phẩn tử là 1
        #x_train_2[:,0] = STT
        x_train_2[:,0] = Re
        x_train_2[:,1] = A
        #x_train_2[:,0] = x_train_2[:,0]
        x_train_2[:,0] = x_train_2[:,0] #/100       # thuan norm gán 
        x_train_2[:,1] = x_train_2[:,1] #/273       # thuan norm        

        #read images to array (empty list)
        #H2_array = []
        P_array = []
        AB_array  = []
        TT_array = []
        #T_array  = []
        U_array  = []
        V_array  = []
        sdf_array= []
      
        for i in range(data_size):
            data_file = '/content/gdrive/MyDrive/Do_an/Interpolation/'+str(Re[i])+'_'+str(A[i])+'.txt'
            #print('reading data file ...',data_file)
            xcor,ycor,Pres,Abpres,Ttpres,UX,UY=load_data(data_file)
            P_array.append(Pres)
            AB_array.append(Abpres)
            TT_array.append(Ttpres)
            U_array.append(UX)
            V_array.append(UY)

        y_train_2 = np.asarray(P_array).astype('float32')#Convert the input to an array(input,data type)
        y_train_3 = np.asarray(AB_array).astype('float32')
        y_train_4 = np.asarray(TT_array).astype('float32')
        y_train_5 = np.asarray(U_array).astype('float32')
        y_train_6 = np.asarray(V_array).astype('float32')
        #Expand the shape of an array
        y_train_2=np.expand_dims(y_train_2,axis=3)
        y_train_3=np.expand_dims(y_train_3,axis=3)
        y_train_4=np.expand_dims(y_train_4,axis=3)
        y_train_5=np.expand_dims(y_train_5,axis=3)
        y_train_6=np.expand_dims(y_train_6,axis=3)     

        #geometry construct - SDF function
        with open('/content/gdrive/MyDrive/Do_an/naca5.csv','r') as csvfile:
            info = csv.reader(csvfile, delimiter=',')
            info=list(info)
        for i in range(5):
            m=float(info[i][0])
            p=float(info[i][1])
            t=float(info[i][2])
            def camber_line(X, m, p, c):
              return np.where(  (X>=0)&(X<=(c*p)),
                                (m/6)*X*( np.power(X/c,2) - 3*p*(X/c) + np.power(p,2)*(3-p)),
                                (m/6)*(np.power(p,3)*(1-(X/c))))
                                
            def dyc_over_dx(X, m, p, c):
              return np.where((X>=0)&(X<=(c*p)),
                              (m/6)*(3*np.power(X/c,2) - 6*p*(X/c) + np.power(p,2)*(3-p)),
                              (m/6)*(-1)*np.power(p,3))


            def thickness(X, t, c):
                term1 =  0.2969 * (np.sqrt(X/c))
                term2 = -0.1260 * (X/c)
                term3 = -0.3516 * np.power(X/c,2)
                term4 =  0.2843 * np.power(X/c,3)
                term5 = -0.1015 * np.power(X/c,4)
                return 5 * t * c * (term1 + term2 + term3 + term4 + term5)

            def naca(X, m, p, t, c=1):
                dyc_dx = dyc_over_dx(X, m, p, c)
                th = np.arctan(dyc_dx)
                yt = thickness(X, t, c)
                yc = camber_line(X, m, p, c)  
                return (yc + yt*np.cos(th),
                        yc - yt*np.cos(th))
            #naca2412            
            c = 1.0
            dx = 2/300
            X, Y = np.meshgrid(np.linspace(-0.5,1.5,300), np.linspace(-1,1,300))

            def Z(X,Y):
              y1= naca(X, m, p, t, c=1)[0]
              y2= naca(X, m, p, t, c=1)[1]
              phi = 1 * np.ones_like(X)
              phi[np.logical_and(Y < y1, Y > y2)] = -1
              return skfmm.distance(phi,dx) 
            d = Z(X,Y)
            fig = plt.figure()                                          
            plt.gca().set_aspect('equal', adjustable='box')             
            plt.imshow(d, vmin=d.min(), vmax=d.max(), origin='lower',  
                        extent=[X.min(), X.max(), Y.min(), Y.max()])          
            fig, ax = plt.subplots()                                    
            plt.gca().set_aspect('equal', adjustable='box')             
            CS = ax.contour(X, Y, d,15) 
            ax.clabel(CS, inline=True, fontsize=10) 
            ax.set_title('SDF value')
            plt.savefig('sdf_value_naca5'+str(i)+'.png')
                  

        with open('/content/gdrive/MyDrive/Do_an/naca4.csv','r') as csvfile:
            info = csv.reader(csvfile, delimiter=',')
            info=list(info)
        for i in range(5):
            m=float(info[i][0])
            p=float(info[i][1])
            t=float(info[i][2])
            def camber_line( X, m, p, c ):
                return np.where((X>=0)&(X<=(c*p)),
                                m * (X / np.power(p,2)) * (2.0 * p - (X / c)),
                                m * ((c - X) / np.power(1-p,2)) * (1.0 + (X / c) - 2.0 * p ))

            def dyc_over_dx( X, m, p, c ):
                return np.where((X>=0)&(X<=(c*p)),
                                ((2.0 * m) / np.power(p,2)) * (p - X / c),
                                ((2.0 * m ) / np.power(1-p,2)) * (p - X / c ))

            def thickness( X, t, c ):
                term1 =  0.2969 * (np.sqrt(X/c))
                term2 = -0.1260 * (X/c)
                term3 = -0.3516 * np.power(X/c,2)
                term4 =  0.2843 * np.power(X/c,3)
                term5 = -0.1015 * np.power(X/c,4)
                return 5 * t * c * (term1 + term2 + term3 + term4 + term5)

            def naca(X, m, p, t, c=1):
                dyc_dx = dyc_over_dx(X, m, p, c)
                th = np.arctan(dyc_dx)
                yt = thickness(X, t, c)
                yc = camber_line(X, m, p, c)  
                return (yc + yt*np.cos(th),
                        yc - yt*np.cos(th))

            #naca2412  
            c = 1.0
            dx = 2/300
            X, Y = np.meshgrid(np.linspace(-0.5,1.5,300), np.linspace(-1,1,300))
            def Z(X,Y):
              y1= naca(X, m, p, t, c=1)[0]
              y2= naca(X, m, p, t, c=1)[1]
              phi = 1 * np.ones_like(X)
              phi[np.logical_and(Y < y1,Y> y2)] = -1
              return skfmm.distance(phi,dx)
            d = Z(X,Y)
            fig = plt.figure()                                          
            plt.gca().set_aspect('equal', adjustable='box')             
            plt.imshow(d, vmin=d.min(), vmax=d.max(), origin='lower',  
                        extent=[X.min(), X.max(), Y.min(), Y.max()])          
            fig, ax = plt.subplots()                                    
            plt.gca().set_aspect('equal', adjustable='box')             
            CS = ax.contour(X, Y, d,15) 
            ax.clabel(CS, inline=True, fontsize=10) 
            ax.set_title('SDF value')
            plt.savefig('sdf_value_naca4'+str(i)+'.png')
                 
  

        for i in range(data_size):
            sdf_array.append(np.array(d))#Nối các hàng khác vào cuối khung này, trả về một đối tượng mới. Các cột không có trong khung này được thêm vào dưới dạng cột mới. 

        sdf_array = np.asarray(sdf_array).astype('float32')#Convert the input to an array(input,data type)     

        x_train_1 = np.asarray(sdf_array).astype('float32')
        x_train_2 = np.asarray(x_train_2).astype('float32')

        x_train_1 =np.expand_dims(x_train_1,axis=3)
        x_train_2 =np.expand_dims(x_train_2,axis=2)     

        print('x1 shape',x_train_1.shape)
        print('x2 shape',x_train_2.shape)
        print('y2 shape',y_train_2.shape,y_train_2.max())
        print('y3 shape',y_train_3.shape,y_train_3.max())
        print('y5 shape',y_train_5.shape,y_train_5.max())

#===============================================================
    
    csv_logger=CSVLogger('training_%d.csv'%(exp_no))#là một lệnh gọi lại truyền kết quả kỷ nguyên tới tệp CSV
    
    checkpoint = ModelCheckpoint('weights.best.hdf5', monitor=user_moni, verbose=1, save_best_only=True, mode='min') #là một lệnh gọi lại để lưu mô hình Keras hoặc trọng lượng mô hình trong quá trình đào tạo, do đó, mô hình hoặc trọng lượng có thể được tải sau để tiếp tục đào tạo từ trạng thái đã lưu.
    print("user_input",user_inp,user_input_channels)
    
    if user_inp=='Y':
            earlystopping=tf.keras.callbacks.EarlyStopping(monitor=user_moni,min_delta=inp_delta,patience=inp_mineps)
            if user_input_channels==3:
                history = model.fit([x_train_1,x_train_2],[y_train_1],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,earlystopping,checkpoint])
            else:
                history = model.fit([x_train_1,x_train_2],[y_train_2,y_train_3,y_train_5],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,earlystopping,checkpoint])
    else:
            if user_input_channels==3:
                history = model.fit([x_train_1,x_train_2],[y_train_1],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,checkpoint])
            else:
                history = model.fit([x_train_1,x_train_2],[y_train_2,y_train_3,y_train_5],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,checkpoint])    
    
    model.save('Network_Expt_%d.h5'%(exp_no))
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig('h2jet_loss.png')  
    return model
    
        
#def model_prediction(exp_no,model,x1,x2):
def model_prediction(exp_no,model):
    
    with open('/content/gdrive/MyDrive/Do_an/naca5.csv','r') as csvfile:
        info = csv.reader(csvfile, delimiter=',')
        info=list(info)
    for i in range(5):
        m=float(info[i][0])
        p=float(info[i][1])
        t=float(info[i][2])
        def camber_line(X, m, p, c):
          return np.where(  (X>=0)&(X<=(c*p)),
                            (m/6)*X*( np.power(X/c,2) - 3*p*(X/c) + np.power(p,2)*(3-p)),
                            (m/6)*(np.power(p,3)*(1-(X/c))))
                            
        def dyc_over_dx(X, m, p, c):
          return np.where((X>=0)&(X<=(c*p)),
                          (m/6)*(3*np.power(X/c,2) - 6*p*(X/c) + np.power(p,2)*(3-p)),
                          (m/6)*(-1)*np.power(p,3))

        def thickness(X, t, c):
            term1 =  0.2969 * (np.sqrt(X/c))
            term2 = -0.1260 * (X/c)
            term3 = -0.3516 * np.power(X/c,2)
            term4 =  0.2843 * np.power(X/c,3)
            term5 = -0.1015 * np.power(X/c,4)
            return 5 * t * c * (term1 + term2 + term3 + term4 + term5)

        def naca(X, m, p, t, c=1):
            dyc_dx = dyc_over_dx(X, m, p, c)
            th = np.arctan(dyc_dx)
            yt = thickness(X, t, c)
            yc = camber_line(X, m, p, c)  
            return (yc + yt*np.cos(th),
                    yc - yt*np.cos(th))
        xmin = -0.5
        xmax = 1.5
        ymin = -1
        ymax = 1
        nx   = 300
        ny   = 300
        dx   = (xmax-xmin)/nx 
        X, Y = np.meshgrid(np.linspace(xmin,xmax,nx), np.linspace(ymin,ymax,ny))  
        def Z(X,Y):
          y1= naca(X, m, p, t, c=1)[0]
          y2= naca(X, m, p, t, c=1)[1]
          phi = 1 * np.ones_like(X)
          phi[np.logical_and(Y < y1, Y > y2)] = -1
          return skfmm.distance(phi,dx)
        d = Z(X,Y)
    with open('/content/gdrive/MyDrive/Do_an/naca4.csv','r') as csvfile:
        info = csv.reader(csvfile, delimiter=',')
        info=list(info)
    for i in range(0,5):
        m=float(info[i][0])
        p=float(info[i][1])
        t=float(info[i][2])
        def camber_line( X, m, p, c ):
            return np.where((X>=0)&(X<=(c*p)),
                            m * (X / np.power(p,2)) * (2.0 * p - (X / c)),
                            m * ((c - X) / np.power(1-p,2)) * (1.0 + (X / c) - 2.0 * p ))

        def dyc_over_dx( X, m, p, c ):
            return np.where((X>=0)&(X<=(c*p)),
                            ((2.0 * m) / np.power(p,2)) * (p - X / c),
                            ((2.0 * m ) / np.power(1-p,2)) * (p - X / c ))

        def thickness( X, t, c ):
            term1 =  0.2969 * (np.sqrt(X/c))
            term2 = -0.1260 * (X/c)
            term3 = -0.3516 * np.power(X/c,2)
            term4 =  0.2843 * np.power(X/c,3)
            term5 = -0.1015 * np.power(X/c,4)
            return 5 * t * c * (term1 + term2 + term3 + term4 + term5)

        def naca(X, m, p, t, c=1):
            dyc_dx = dyc_over_dx(X, m, p, c)
            th = np.arctan(dyc_dx)
            yt = thickness(X, t, c)
            yc = camber_line(X, m, p, c)  
            return (yc + yt*np.cos(th),
                    yc - yt*np.cos(th))

        xmin = -0.5
        xmax = 1.5
        ymin = -1
        ymax = 1
        nx   = 300
        ny   = 300
        dx   = (xmax-xmin)/nx 
        X, Y = np.meshgrid(np.linspace(xmin,xmax,nx), np.linspace(ymin,ymax,ny)) 
        def Z(X,Y):
          y1= naca(X, m, p, t, c=1)[0]
          y2= naca(X, m, p, t, c=1)[1]
          phi = 1 * np.ones_like(X)
          phi[np.logical_and(Y < y1,Y> y2)] = -1
          return skfmm.distance(phi,dx)
        d = Z(X,Y)
        sdf_array = []

        for i in range(1):
            sdf_array.append(np.array(d))

        x_test_1 = np.asarray(sdf_array).astype('float32')
        x_test_2 = []

        data = read_csv("/content/gdrive/MyDrive/Do_an/Interpolation/labeltest12.csv")
        test_size=int(1)

    # converting column data to list
        Re = data['Reynold'].tolist()
        A = data['AoA'].tolist()

        x_test_2 = np.ones((test_size,2))
        x_test_2[:,0] = Re
        x_test_2[:,1] = A
        x_test_2 = np.asarray(x_test_2).astype('float32')
        x_test_2[:,0] = x_test_2[:,0] #/100
        x_test_2[:,1] = x_test_2[:,1] #/273
        print('x1 shape:',x_test_1.shape)
        print('x2 shape:',x_test_2.shape)

        ans=model.predict([x_test_1,x_test_2])

        p = []
        u  = []
        v  = []

        p_error  = []
        u_error  = []    
        v_error = []

        P_array = []
        U_array = []
        V_array = []

        p = np.squeeze(ans[0], axis=0)
        p = np.squeeze(p, axis=2)
        #t = ans[:,:,1]
        p = p*100000

        u = np.squeeze(ans[1], axis=0)
        u = np.squeeze(u, axis=2)
        #u= ans[:,:,2]
        u = u*100

        v = np.squeeze(ans[2], axis=0)
        v = np.squeeze(v, axis=2)
        #u= ans[:,:,2]
        v = v*100        
    
        print('pmin pmax',p.min(),p.max())
        print('umin umax',u.min(),u.max())
        print('vmin vmax',v.min(),v.max())

        data_file = './content/gdrive/MyDrive/Do_an/12.txt'
        print('reading data file ...',data_file)
        xcor,ycor,Pres,Abpres,Ttpres,UX,UY=load_data(data_file)
        P_array = np.asarray(Pres*100000)
        U_array = np.asarray(UX*100)
        V_array = np.asarray(UY*100)
        def camber_line( X, m, p, c ):
            return np.where((X>=0)&(X<=(c*p)),
                            m * (X / np.power(p,2)) * (2.0 * p - (X / c)),
                            m * ((c - X) / np.power(1-p,2)) * (1.0 + (X / c) - 2.0 * p ))

        def dyc_over_dx( X, m, p, c ):
            return np.where((X>=0)&(X<=(c*p)),
                            ((2.0 * m) / np.power(p,2)) * (p - X / c),
                            ((2.0 * m ) / np.power(1-p,2)) * (p - X / c ))

        def thickness( X, t, c ):
            term1 =  0.2969 * (np.sqrt(X/c))
            term2 = -0.1260 * (X/c)
            term3 = -0.3516 * np.power(X/c,2)
            term4 =  0.2843 * np.power(X/c,3)
            term5 = -0.1015 * np.power(X/c,4)
            return 5 * t * c * (term1 + term2 + term3 + term4 + term5)

        def naca1(X, m, p, t, c=1):
            dyc_dx = dyc_over_dx(X, m, p, c)
            th = np.arctan(dyc_dx)
            yt = thickness(X, t, c)
            yc = camber_line(X, m, p, c)  
            return  yc + yt*np.cos(th)

        def naca2(X, m, p, t, c=1):
            dyc_dx = dyc_over_dx(X, m, p, c)
            th = np.arctan(dyc_dx)
            yt = thickness(X, t, c)
            yc = camber_line(X, m, p, c)  
            return  yc - yt*np.cos(th)
            #naca2412  
        c = 1.0
        m = 0.02    
        p = 0.4
        t = 0.12 
        X, Y = np.meshgrid(np.linspace(-0.5,1.5,300), np.linspace(-1,1,300))
        y1= naca1(X, m, p, t, c=1)
        y2= naca2(X, m, p, t, c=1)
        P_array[np.logical_and(Y <= y1,Y>= y2)] = 0
        

        plot_image(P_array,'CFD_','Pressure',1)
        plot_image(U_array,'CFD_','Velocity',2)
        plot_image(V_array,'CFD_','Velocity',3)

        plot_image(p,'Predict_','Pressure',1)
        plot_image(u,'Predict_','Velocity',2)
        plot_image(v,'Predict_','Velocity',3)

    
        #giá trị tyêt đối
        
        p_error  = abs(P_array-p)
        u_error  = abs(U_array-u)
        v_error  = abs(V_array-v)

        plot_image(p_error,'error_abs_','Pressure',1)   
        plot_image(u_error,'error_abs_','Velocity',2)
        plot_image(v_error,'error_abs_','Velocity',3) 

        pmax = P_array.max()
        umax = U_array.max()
        vmax = V_array.max()
        for j in range(ny):
            for i in range(nx):
                p_error[i,j]  = abs(P_array [i,j]-p [i,j])*100/pmax
                u_error[i,j]  = abs(U_array [i,j]-u [i,j])*100/umax
                v_error[i,j]  = abs(V_array [i,j]-v [i,j])*100/vmax

        plot_image(p_error,'error_rel_','Pressure',1)   
        plot_image(u_error,'error_rel_','Velocity', 2)
        plot_image(v_error,'error_rel_','Velocity', 3) 

        return ans

def save_model(model,modelname):
    model.save(modelname+'.h5')
    return


def load(modelname):
    model=load_model(modelname)
    return model

def plot_image(var,pretext,fieldname,flag):
    if (flag==1):
        labeltxt = 'Pressure (Pa)'
    elif (flag==2):
        labeltxt = 'X Velocity (m/s)'
    elif (flag==3):
        labeltxt = 'Y Velocity (m/s)'
        

    X, Y = np.meshgrid(np.linspace(-0.5,1.5,300), np.linspace(-1,1,300))
    fig = plt.figure()

    im = plt.imshow(var, vmin=var.min(), vmax=var.max(), origin='lower',
            extent=[X.min(), X.max(), Y.min(), Y.max()])

    #fig, ax = plt.subplots()#tạo hình
    fig = plt.subplots()
    plt.gca().set_aspect('equal', adjustable='box')     ## square image
    plt.contourf(X, Y, var,50,cmap=plt.cm.rainbow)
    plt.colorbar(label=labeltxt) 
    plt.savefig(pretext+fieldname+'.png')  
    plt.close()

