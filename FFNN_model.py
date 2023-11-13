#%%########################################## LOADING PYTHON PACKAGES ###############################################
import os, sys, psutil
directory = os.environ['directory']
sys.path.append(directory)

import glob
import datetime,time
from datetime import timedelta
from netCDF4 import Dataset, num2date, date2num
import numpy as np
import random
import keras.callbacks
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop

#%%
direct_in = directory + '/Data/' 


month_MLP = 6    # Reconstruction month
year_MLP  = 2021 # Reconstruction year

if month_MLP<=9:
    str_month = str(0) + str(month_MLP)
else:
    str_month =  str(month_MLP)
str_year = str(year_MLP)

t = (year_MLP-1985)*12 + month_MLP-1

#%%############################################### LOADING INPUT DATA #################################################
direct_train = direct_in+'Data_model_train_r100_1985to2021_05to07.nc'
ind_months = Dataset(direct_train)['idmonth'][:].astype('int')
ind_taken        = np.where(ind_months!=t) #Remove data in the reconstruction month
data_train_all   = Dataset(direct_train)['data_train'][:] 
fuCO2_SOCAT      = data_train_all[:,0].squeeze() 
fuCO2_Taka_SOCAT = data_train_all[:,1].squeeze() 
SSS_SOCAT        = data_train_all[:,2].squeeze()
SST_SOCAT        = data_train_all[:,3].squeeze()
SSH_SOCAT        = data_train_all[:,4].squeeze()
CHL_SOCAT        = data_train_all[:,5].squeeze() 
MLD_SOCAT        = data_train_all[:,6].squeeze()
aCO2_SOCAT       = data_train_all[:,7].squeeze() 
lat_train        = data_train_all[:,8].squeeze()[ind_taken].squeeze()
lon_train1       = data_train_all[:,9].squeeze()[ind_taken].squeeze()
lon_train2       = data_train_all[:,10].squeeze()[ind_taken].squeeze()
#transferi of CHL and MLD to log 
CHL_log = np.log(CHL_SOCAT) 
MLD_log = np.log(MLD_SOCAT) 

std_SOCAT  = np.nanstd(fuCO2_SOCAT) 
mean_SOCAT = np.nanmean(fuCO2_SOCAT)

#!!!!!scaling of data used as predictors in training algorithm!!!!!!!!!!!
SSS_train       = (SSS_SOCAT[ind_taken].squeeze() - np.nanmean(SSS_SOCAT))/np.nanstd(SSS_SOCAT)
SST_train       = (SST_SOCAT[ind_taken].squeeze() - np.nanmean(SST_SOCAT))/np.nanstd(SST_SOCAT)
SSH_train       = (SSH_SOCAT[ind_taken].squeeze() - np.nanmean(SSH_SOCAT))/np.nanstd(SSH_SOCAT)
CHL_train       = (CHL_log[ind_taken].squeeze() - np.nanmean(CHL_log))/np.nanstd(CHL_log)
MLD_train       = (MLD_log[ind_taken].squeeze() - np.nanmean(MLD_log))/np.nanstd(MLD_log)
aCO2_train      = (aCO2_SOCAT[ind_taken].squeeze() - np.nanmean(aCO2_SOCAT))/np.nanstd(aCO2_SOCAT)
fuCO2_Taka_train = (fuCO2_Taka_SOCAT[ind_taken].squeeze()  - np.nanmean(fuCO2_Taka_SOCAT ))/np.nanstd(fuCO2_Taka_SOCAT )
fuCO2_train      = (fuCO2_SOCAT[ind_taken].squeeze()  - np.nanmean(fuCO2_SOCAT ))/np.nanstd(fuCO2_SOCAT)

sum_data = fuCO2_train + fuCO2_Taka_train + SSS_train + SST_train + SSH_train + CHL_train + MLD_train+ aCO2_train #
ind_obs_train = np.where(~np.isnan(sum_data))

data_predictors = np.column_stack((fuCO2_Taka_train[ind_obs_train], SSS_train[ind_obs_train], SST_train[ind_obs_train], SSH_train[ind_obs_train],CHL_train[ind_obs_train], MLD_train[ind_obs_train], aCO2_train[ind_obs_train],lat_train[ind_obs_train],lon_train1[ind_obs_train],lon_train2[ind_obs_train]))  #matrix of preditors that will be used in training algorithm
fuCO2_list = fuCO2_train[ind_obs_train] 


np.random.seed();# random number generator : repeat M times to get M samples of reconstructed fCO2
N =np.size(data_predictors,axis=0); N_train = int(2*N/3) #of data chosen randomly for training
index1=np.sort(np.random.choice(np.arange(0,N,dtype=int), N_train, replace=False))

data_train = data_predictors[index1,...]
fuCO2_list_train =  fuCO2_list[index1] 
data_val =  np.delete(data_predictors,index1,axis=0)  #data for validation (add to validate a model during the iterative procedure of training)
fuCO2_list_val =  np.delete(fuCO2_list,index1)        #data for validation (add to validate a model during the iterative procedure of training)
nbpredictors = np.shape(data_predictors)[1]

#%%############################################### FFNN MODEL DESIGN #################################################
nbround = 2   # number of decimals for model statistics
nbepoch = 500 # number of epoche for model early stopping
bsize   = 32  # batch size depending on training data length

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(0.05*nbepoch), verbose=0, mode='auto') 
model = Sequential() #Model architech
model.add(Dense( int(nbpredictors*1.5), input_dim=nbpredictors , kernel_initializer='glorot_uniform')) 
model.add(Activation('tanh'))
model.add(Dense(nbpredictors, kernel_initializer='glorot_uniform'))
model.add(Activation('tanh'))
model.add(Dense(1, kernel_initializer='glorot_uniform'))
model.add(Activation('linear'))
sgd = SGD(lr=0.001, decay=0.0, momentum=0.0, nesterov=False)   # stochastic gradient descent optimizer. Includes support for momentum, learning rate decay, and Nesterov momentum.
rmsprop = RMSprop(lr=0.01, rho=0.95, epsilon=1e-08, decay=0.0) # freely tuned optimizer.
L = model.compile(loss='mse',optimizer=rmsprop)
H = model.fit(data_train, fuCO2_list_train, epochs=nbepoch,callbacks=[earlyStopping], verbose=0, batch_size=bsize, validation_data=(data_val,fuCO2_list_val))


#TRAINING EVALUATION

score  =  round(np.sqrt(model.evaluate(data_train, fuCO2_list_train, batch_size=bsize))* std_SOCAT,nbround)
scoreVAL  =  round(np.sqrt(model.evaluate(data_val, fuCO2_list_val, batch_size= bsize))* std_SOCAT,nbround)

str_score = str(score)
while (len(str_score) < nbround+3):
    for istrsc in range(len(str_score), nbround+3):
        str_score += '0'
str_scoreVAL = str(scoreVAL)
while (len(str_scoreVAL) < nbround+3):
    for istrsc in range(len(str_scoreVAL), nbround+3):
        str_scoreVAL += '0' 
  
print('Ndata - ', 'Train: ', N_train, 'VAL: ', N-N_train)
print('scoreTRAIN: ', score,'scoreVAL: ', scoreVAL)

#%%#################################### FFNN MODEL ESTIMATION and EVALUATION #########################################
direct_SOCAT = direct_in+'SOCAT_r100_202106.nc' # SOCAT data for reconstruction evaluation
lon_SOCAT = Dataset(direct_SOCAT)['lon'][:]
lat_SOCAT = Dataset(direct_SOCAT)['lat'][:]
fuCO2_SOCAT_reconstr = Dataset(direct_SOCAT)['fuCO2'][:].squeeze() 

#read predictor data for fuCO2 reconstruction (prediction) over the global ocean
direct_SSS       = direct_in+'SSS_r100_202106.nc'
direct_SST       = direct_in+'SST_r100_202106.nc'
direct_SSH       = direct_in+'SSH_r100_202106.nc'
direct_CHL       = direct_in+'CHL_r100_202106.nc'
direct_MLD       = direct_in+'MLD_r100_202106.nc'
direct_fuCO2clim = direct_in+'fuCO2_clim_r100_06.nc'
direct_xCO2      = direct_in+'xCO2_r100_202106.nc'

SSS         = Dataset(direct_SSS)['SSS'][:].squeeze()
SST         = Dataset(direct_SST)['SST'][:].squeeze()
SSH         = Dataset(direct_SSH)['SSH'][:].squeeze()
CHL         = Dataset(direct_CHL)['CHL'][:].squeeze()
MLD         = Dataset(direct_MLD)['MLD'][:].squeeze()
fuCO2_Taka  =  Dataset(direct_fuCO2clim)['fuCO2'][:].squeeze() 
aCO2        = Dataset(direct_xCO2)['CO2'][:].squeeze()

#!!!!!scaling of data used in reconstruction (prediction)!!!!!
all_SSS_predict = (SSS - np.nanmean(SSS_SOCAT))/np.nanstd(SSS_SOCAT) 
all_SST_predict = (SST - np.nanmean(SST_SOCAT))/np.nanstd(SST_SOCAT)
all_SSH_predict = (SSH - np.nanmean(SSH_SOCAT))/np.nanstd(SSH_SOCAT)
all_CHL_predict = (CHL_log_predict - np.nanmean(CHL_log))/np.nanstd(CHL_log)
all_MLD_predict = (MLD_log_predict - np.nanmean(MLD_log))/np.nanstd(MLD_log)
all_aCO2_predict  = (aCO2 - np.nanmean(aCO2_SOCAT))/np.nanstd(aCO2_SOCAT)
all_fuCO2_Taka_predict = (fuCO2_Taka   - np.nanmean(fuCO2_Taka_SOCAT ))/np.nanstd(fuCO2_Taka_SOCAT)

sum_predict = all_SSS_predict + all_SST_predict +  all_SSH_predict +all_CHL_predict + all_MLD_predict + all_aCO2_predict + all_fuCO2_Taka_predict
ind_obs = np.where(~np.isnan(sum_predict))
SSS_predict_list =  all_SSS_predict[ind_obs]
SST_predict_list =  all_SST_predict[ind_obs]
SSH_predict_list =  all_SSH_predict[ind_obs]
CHL_predict_list =  all_CHL_predict[ind_obs]
MLD_predict_list =  all_MLD_predict[ind_obs]
aCO2_predict_list = all_aCO2_predict[ind_obs]
fuCO2_Taka_predict_list = all_fuCO2_Taka_predict[ind_obs]

lat_predict_list =  np.sin(lat_SOCAT[ind_obs[0]]*np.pi/180.)
lon_predict_list1 = np.cos(lon_SOCAT[ind_obs[1]]*np.pi/180.)
lon_predict_list2 = np.sin(lon_SOCAT[ind_obs[1]]*np.pi/180.) 

data_reconstr = np.column_stack(( fuCO2_Taka_predict_list, SSS_predict_list,SST_predict_list,SSH_predict_list,
                                  CHL_predict_list,MLD_predict_list,aCO2_predict_list,
                                  lat_predict_list,lon_predict_list1,lon_predict_list2))


fuCO2_matrix = np.nan*np.zeros((len(lat_SOCAT),len(lon_SOCAT)))
preds1 = model.predict(data_reconstr)
#     print('Size of data_reconst= {}, Size of preds1= {}'.format(data_reconstr.shape, preds1.shape))
preds_list = preds1[:,0]

fuCO2_matrix[ind_obs] = preds_list * std_SOCAT + mean_SOCAT


#RECONSTRUCTION (PREDICTION) EVALUATION
ind_obseval = np.where(~np.isnan(fuCO2_matrix + fuCO2_SOCAT_reconstr)) 
scoreEVAL  = round(np.sqrt(np.nanmean((fuCO2_matrix[ind_obseval]- fuCO2_SOCAT_reconstr[ind_obseval])**2)) ,nbround)
str_scoreEVAL = str(scoreEVAL)        
while (len(str_scoreEVAL) < nbround+3):
    for istrsc in range(len(str_scoreEVAL), nbround+3):
        str_scoreEVAL += '0'
print('RMSD of Estimated fuCO2: ', 'scoreEVAL: ', scoreEVAL)