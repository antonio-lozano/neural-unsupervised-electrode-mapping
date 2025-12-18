# -*- coding: utf-8 -*-
"""
Utility functions for data processing, cleaning, plotting etc

@author: Antonio Lozano
"""
import time 
import numpy as np
import pickle
import hdbscan
import pandas as pd

import umap
import scipy
from scipy.io import loadmat
from scipy.linalg import orthogonal_procrustes
import glob
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from sklearn.metrics import euclidean_distances
from scipy.stats import ttest_ind
import os
from sklearn.decomposition import FastICA, PCA
from sklearn.metrics import mean_squared_error
from numpy.random import default_rng
import seaborn as sns


#Colors for 16 arrays in L
colsL = [(1.0000, 0,  0,), (1.0000, 0.3750, 0), (1.0000, 0.7500, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706, 0.0745),
         (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0, 0.2500, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750)]


def load_SNR_instances_openEyes(folder_path):
    """
    Load oad SNR instances calculated from LFP responses. Instances are called 'mean_MUA_instanceX_new.mat'

    SNR_list will contain all the 8 instances of SNR. Each instance contains a list of chunks
    of data where the eyes were closed
    """
    SNR_list = []

    # try:
    for i in range(8):
        
        try:
            SNR_list.append(scipy.io.loadmat(folder_path + 'mean_MUA_instance' + str(i+1) + '_new.mat')['channelSNR'][0])   

        except:
            print('Error loading lfp instance ', i+1)
        
    return SNR_list


def order_SNR_instances(snrList, arrayNums, channelNums, V1ONLY=False):
    """
    Helper function
    """
    # if V1ONLY:
    #     toTake = np.hstack((np.linspace(0,63,64), np.linspace(192,1023,1024-192))).astype(int)
    # else:
    #     toTake =np.linspace(0,1023,1024).astype(int)
    
    orderedSNRs = None
    snrArrs = []
    
    # for each of the 8 instances, order the arrays
    for insC in range(snrList.shape[0]):
        numeritos = arrayNums[:,insC]
        
        # find which arrays are in each instance
        whichArr = np.unique(numeritos)
        
        # for each of the arrays, get the data and order it
        for w in whichArr:
            # where are the arrays channels
            donde = np.where(numeritos == w)
            
            # getting rfs for that array
            SNRArr = snrList[insC][donde]
            
            # what is their real index of those arrays
            canalesInd = channelNums[:,insC][donde] - 1
            SNRArr = SNRArr[canalesInd]
                    
            if orderedSNRs is None: 
                orderedSNRs = SNRArr
    
            else:
                orderedSNRs = np.vstack((orderedSNRs,SNRArr))
                
            snrArrs.append(SNRArr)
            
    snrArrs     = np.asarray(snrArrs,dtype='float32')    
    orderedSNRs = np.asarray(orderedSNRs,dtype='float32')    
     
    return snrArrs, orderedSNRs


def load_MUA_instances(folder_path, chunks = [0,16,33]):
    """
    lfp_list will contain all the 8 instances of LFP. Each instance contains a list of chunks
    of data where the eyes were closed
    """
    lfp_list = []
    
    try:
        # This works for monkey_L, the MUA file name is different
        for i in range(8):
            # try:
                mua = scipy.io.loadmat(folder_path + 'MUA_eyes_closed_instance' + str(i+1) + '.mat')['MUA_eyes_closed_data'][0]
                # print('This function loads a by HAND selection of MUA instances: 0, 16, 33')
                mua = mua[chunks]
                print('Loading MUA chunk ', i+1)      
                print(mua[0].shape)
                lfp_list.append(mua)
            # except:
            #     print('Error loading MUA instance ', i+1)
    except:
        #This works for monkey_A, the MUA file name is different
        print('Loading monkey_A MUA')
        for i in range(8):
            # try:
                
                mua = scipy.io.loadmat(folder_path + 'MUA_eyes_closed_NSP' + str(i+1) + '.mat')['MUA_eyes_closed_data'][0]
                # print('This function loads a by HAND selection of MUA instances: 0, 16, 33')
                mua = mua[chunks]
                print('Loading MUA chunk ', i+1)      
                print(mua[0].shape)
                lfp_list.append(mua)
        
    return lfp_list


def load_MUA_instances_ALL(folder_path):
    """
    LOADING ALL MUA instances
    
    lfp_list will contain all the 8 instances of LFP. Each instance contains a list of chunks
    of data where the eyes were closed
    """
    lfp_list = []
    
    try:
        # This works for monkey_L, the MUA file name is different (INSTANCE)
        for i in range(8):
            # try:
                mua = scipy.io.loadmat(folder_path + 'MUA_eyes_closed_instance' + str(i+1) + '.mat')['MUA_eyes_closed_data'][0]
                # print('This function loads a by HAND selection of MUA instances: 0, 16, 33')
                #mua = mua[chunks]
                print('Loading MUA chunk ', i+1)      
                # print(mua[0].shape)
                lfp_list.append(mua)
            # except:
            #     print('Error loading MUA instance ', i+1)
    except:
        #This works for monkey_A, the MUA file name is different (NSP)
        print('Loading monkey_A MUA')
        for i in range(8):
            # try:
                
                mua = scipy.io.loadmat(folder_path + 'MUA_eyes_closed_instance' + str(i+1) + '.mat')['MUA_eyes_closed_data'][0]
                # print('This function loads a by HAND selection of MUA instances: 0, 16, 33')
                #mua = mua[chunks]
                print('Loading MUA chunk ', i+1)      
                # print(mua[0].shape)
                lfp_list.append(mua)
        
    return lfp_list

 
def load_MUA_instances_openEyes(folder_path, chunks = [0,16,33]):
    """
    lfp_list will contain all the 8 instances of LFP. Each instance contains a list of chunks
    of data where the eyes were closed
    """
    lfp_list = []
    

    # This works for monkey_L, the MUA file name is different
    for i in range(8):
        try:
            # # print('This function loads a by HAND selection of MUA instances: 0, 16, 33')
            print('Loading MUA chunk ', i+1)      

            ########
            ins = scipy.io.loadmat(folder_path + 'MUA_instance' + str(i+1) + '.mat')['channelDataMUA'][0]
    
            #############################
            seconds = 0.3
            freq_sampling = 1000
            # calculating the amount of data samples to take
            # to get only spontaneous data
            
            dataInterval = int( seconds * freq_sampling )
            chan_dataList = []
            for channel in range(128):

                # going through each channel's data 
                # (each data containing many trials)
                chan_data = ins[channel]
                
                # using only the first 300 ms of data
                chan_data = chan_data[:, 0 : dataInterval]
                
                # appending data
                chan_dataList.append(np.hstack(chan_data))
            
            ins = np.asarray(chan_dataList)
            ###############################
            
            lfp_list.append(ins)

            ##############

        except:
            print('Error loading MUA instance ', i+1)
            
            
    # this is a fix so the data has the correct dimensions to order it
    aux = []
    for i in range(len(lfp_list)): 
        aux.append(np.expand_dims(lfp_list[i],0))
    lfp_list = aux
    
    return lfp_list


def load_MUA_3bin_instances(folder_path, chunks = [0,16,33]):
    """
    lfp_list will contain all the 8 instances of LFP. Each instance contains a list of chunks
    of data where the eyes were closed
    """
    lfp_list = []
    for i in range(8):
        # try:
            mua = scipy.io.loadmat(folder_path + 'MUA_eyes_closed_instance' + str(i+1) + '_3bins.mat')['MUA_eyes_closed_data_3bins'][0]
            # print('This function loads a by HAND selection of MUA instances: 0, 16, 33')
            mua = mua[chunks]
            
            print(mua[0].shape)
            lfp_list.append(mua)
        # except:
        #     print('Error loading MUA instance ', i+1)
    return lfp_list


def bin_MUA(data,step):
    """
    bin MUA (averaging by dividing by the step size)
    divideByMax = True means dividing each channel by its max value
    """

    muaBinned = []
    for i in range(data.shape[0]):
        
        m = data[i]
        mBins = []
    
        bins = np.arange(0,m.shape[0],step)
        for j in range(bins.shape[0]):
            
                # averaging the MUA for each bin
                aux = m[bins[j]:bins[j]+step].sum() / step
                mBins.append(aux)
                
        mBins = np.asarray(mBins,dtype='float32')
        muaBinned.append(mBins)
        
    muaBinned = np.asarray(muaBinned,dtype='float32')
    return muaBinned


def bin_and_norm(data,binsize):
    """ binning and normalization"""
    
    MUA0 = bin_MUA(data,binsize)
    me = np.expand_dims(MUA0.mean(1),1)
    st = np.expand_dims(MUA0.std(1),1)
    MUA0 =  ( MUA0 - me ) / st
    
    return MUA0


def compare_binnings(MUA,step = [1,10,100]):
    """
    compare binnings
    """
    for j in range(len(step)):
        plt.figure(figsize=(20,20))
        contador = 0
    
        hlim = int(4000/step[j])
        
        print(f'hlim {hlim}')
        
        for i in range(50):
            toPlot = MUA[j][contador][0:hlim]
            #toPlot /= toPlot.max()
            plt.plot( toPlot + i * 20, color = 'black', linewidth=4)
            contador = contador + 10
        plt.title('MUA')
          
def load_valid_utahLocations(electrodepixelsPATH= 
                             r'LFP-RFs\data\coordinates_of_electrodes_on_cortex_using_photos_of_arrays\allPixelIDs_monkey_L.mat',
                             toDelete=None):
    """
    Loading of coordinates.
    """
    utah = loadmat(electrodepixelsPATH)
    
    x_coords = utah['allMiddleXCoordsGrid']
    y_coords = utah['allMiddleYCoordsGrid']
    
    num_arrays = x_coords.shape[2]
    num_electrodes = x_coords.shape[0] * x_coords.shape[1]
    utahNew = np.array([x_coords,-y_coords]).T
    
    utahNew = utahNew.reshape(num_arrays*num_electrodes,2)
    
    if toDelete is not None:
        print('Deleting some custom channels ', str(toDelete))
        utahNew = np.delete(utahNew, toDelete, axis = 0)        
    
    return utahNew
    

utah = load_valid_utahLocations()

def load_snr(snrPath = r'cross_talk_removal\low_SNR_electrodes_L_RS_090817.txt'):

    return np.loadtxt( snrPath ).astype('int64') - 1

def load_old_snr(allsnrPATH= 'LFP-RFs/allsnr.p'): 
    """
    Load Signal to Noise Ratios for monkey_L
    """

    
    allsnr = pickle.load(open(allsnrPATH,'rb'))
    idxSNR = np.where( allsnr < 2 )[0]
    
    return idxSNR

def load_lowSNRelect_Aitor(path = r"cross_talk_removal\L_RS_090817_removal_metadata.csv"):

    """
    """
    file1 = open(path,"r")
    
    lowSNR = []
    
    print('Warning: loading crosstalk electrodes, we correct the ID = ID - 1 because of pythons 0 indexing')
    
    for line in file1:
    
        l = line.split(',')[0] 
        
        # ignoring first line
        if l != 'Electrode_ID':
            
            electrode = line.split(',')[0]
            lowSNR.append(np.int64(float(electrode)))
            
    return np.asarray(lowSNR,'int64') - 1
            

def load_crosstalk_electrodes(path = r"cross_talk_removal\L_RS_090817_removal_metadata.csv"):
    
    """in this new version I do (np.int64(float(l))) instead of (np.int64(l))
    
    
    """
    file1 = open(path,"r")
    
    crossTalkers = []
    
    print('Warning: loading crosstalk electrodes, we correct the ID = ID - 1 because of pythons 0 indexing')
    
    for line in file1:

        l = line.split(',')[-2]
        #print(l)

        if l != 'Removed electrode ID':# we don't want the first row
            try:
                crossTalkers.append(np.int64(float(l)))
                #print('APPENDING L ')
            except:
                print('Something didnt work while appending line for crosstalk electrode ID')
        else:
            None
            
    # print(crossTalkers[0])    
    return np.asarray(crossTalkers,'int64') - 1


def load_crosstalk_electrodes_old(path = r"cross_talk_removal\L_RS_090817_removal_metadata.csv"):
    file1 = open(path,"r")
    
    crossTalkers = []
    
    print('Warning: loading crosstalk electrodes, we correct the ID = ID - 1 because of pythons 0 indexing')
    
    for line in file1:

        l = line.split(',')[-2]
        #print(l)
        if l != 'Removed electrode ID':# we don't want the first row
            try:
                crossTalkers.append(np.int64(l))
                # print('APPENDING L ')
            except:
                print('Something didnt work while appending line for crosstalk electrode ID')
        else:
            None
            
    # print(crossTalkers[0])    
    return np.asarray(crossTalkers,'int64') - 1

def load_valid_rfs(rfilePATH= 'rfile.p',
                   allsnrPATH= 'allsnr.p',
                   outlierRejectionQuantile= 0.95,
                   deleteCustomChannels=np.array([None]),
                   allow_single_cluster=True):
    
    """
    Input: paths to receptive field and snr calculations pickles
    Outputs = receptive field locations and deleted electrode indexes
    """
    
    print('Loading receptive fields and deleting outliers and low SNR channels')
    # loading rfs
    rfile=pickle.load(open(rfilePATH,'rb'))
    
    # creating rf matrix from dictionary
    rfList = []
    for k in rfile.keys():
        rfList.append(rfile[k][:,:])
        
    rfs = np.array(rfList).astype('float32')[:,:,0:2]
    
    num_arrays = rfs.shape[0]
    num_electrodes = rfs.shape[1]
    
    # loading signal to noise ratios
    print('WARNING at function load_valid_rfs: ')
    print('Not loading allSNR anymore from allsnrPATH')
    print('Instead, provide the bad channels directly at the argument deleteCustomChannels')

    # allsnr = pickle.load(open(allsnrPATH,'rb'))
    # idxSNR = np.where( allsnr < 2 )[0]
    
    # reshaping array to work more comfy
    rfs = rfs.reshape(num_arrays*num_electrodes,2)
    
    # Now, we will find RF ouliers
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, allow_single_cluster=allow_single_cluster).fit(rfs)
    
    clusterer.outlier_scores_
    # sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
    
    threshold = pd.Series(clusterer.outlier_scores_).quantile(outlierRejectionQuantile)
    outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
    
    print('Outliers = ', outliers)
    # es_toDelete = np.union1d(outliers, idxSNR)
    es_toDelete = outliers

    if deleteCustomChannels.any() != None:
        # print('Deleting also some custom channels ', str(deleteCustomChannels))
        es_toDelete = np.union1d(es_toDelete, deleteCustomChannels)
        
    # deleting rfs
    rfs = np.delete(rfs, es_toDelete, axis = 0)
    
    #deleting array peretenence index
    # plotting
    # plt.figure(figsize=(20,20))
    # plt.scatter(rfs[:,0],rfs[:,1] , s=50, linewidth=0, alpha=0.9)
    
    print('Returning RFs and the list of deleted electrodes')
    return rfs, es_toDelete


def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, 
              metric='correlation', title='', 
              c=None, text=False, drawCornersOnly64=False, electrodesNumber=96):
    """
    draw umap
    """
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    # u = fit.fit_transform(data);
    mapper = fit.fit(data)
    
    u = fit.transform(data)
    
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=c)
        for i in range(u.shape[0]):
            ax.text(u[i,0] * (1 + 0.01), u[i,1] * (1 + 0.01) , i, fontsize=15)
            
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=c, s=100)
        
        #### TEXT
        if text == True:
            
            print("drawCornersOnly64 ", drawCornersOnly64)
            if drawCornersOnly64 == True:
            
                #Draw corners AND polygons
                counter = 0
                arrayCounter = 0
                polyList = []
                cornersList = []
                for i in range(u.shape[0]):
                    
                    if counter == 0 or counter == 7 or counter == 55 or counter == 63:
                        ax.text(u[i,0] * (1 + 0.01), u[i,1] * (1 + 0.01) , counter + 1, fontsize=20)
                        
                        #Creating a list of corners for the polygon
                        cornersList.append(([u[i,0],u[i,1]]))
                        
                    counter = counter + 1
                        
                    if counter == electrodesNumber:
                        
                        arrayCounter = arrayCounter + 1      
                        
                        #creating polygons from the array's edges 
                        cornersList = np.array(cornersList)
                        
                        polygon = Polygon(cornersList, True)
                        polyList.append(polygon)
                        
                        cornersList = []
                        counter = 0  

                # #Creating polygon drawings
                p = PatchCollection(polyList, cmap=matplotlib.cm.gray, alpha=0.5)
                colors = np.ones(len(polyList))
                p.set_array(np.array(colors))
                ax.add_collection(p)
                #ax.plot(umap.plot.connectivity(fit, show_points=True))

            else:
                counter = 0
                for i in range(u.shape[0]):
                    
                    ax.text(u[i,0] * (1 + 0.01), u[i,1] * (1 + 0.01) , counter + 1, fontsize=20)
                    counter = counter + 1
                    
                    if counter == electrodesNumber:
                        counter = 0    

    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=c, s=100)
        
        if text == True:
            counter = 0
            for i in range(u.shape[0]):
                
                ax.text(u[i,0] * (1 + 0.01), u[i,1] * (1 + 0.01) ,  u[i,2] * (1 + 0.01) ,counter + 1, fontsize=20)
                counter = counter + 1
                
                if counter == electrodesNumber:
                    counter = 0
    plt.title(title, fontsize=18)

    return fit


def separate_rf_utah_lfp(DATA, rfs, utah, numbers):
    """
    DATA = lfp
    numbers = array numeration [1,1,1,2,2,3,3,3,3,3,4,4,9,9,....16,16]
    
    """
    rfs_arrays = []
    utah_arrays = []
    lfp_arrays   = []
    uniqueArrays = np.unique(numbers)
    
    for i in range(len(uniqueArrays)):
        
        print('Getting array ' + str(i+1))
        indices = np.where(numbers == int(uniqueArrays[i]))[0]
        
        lfp   = DATA[indices]
        x_rfs = rfs[indices,0]
        y_rfs = rfs[indices,1]
        
        x_utah = utah[indices,0]
        y_utah = utah[indices,1]
    
        utah_arrays.append(np.array([x_utah, y_utah]).T)
        rfs_arrays.append(np.array([x_rfs, y_rfs]).T)
        lfp_arrays.append(lfp)
        
    return rfs_arrays, utah_arrays, lfp_arrays


def separate_arrays(raw):
    
    print('WARNING, THIS ONLY WORKS PROPERLY FOR THE 1024 ARRAY, NOT IF IT HAS LESS CHANNELS')
    arrayList = []
    
    for i in np.arange(0,1024,64):
    
        array = raw[i:i+64,:]
        arrayList.append(array)

    return arrayList


def separate_arrays_fromIndex(raw, arrayNumber, num_anchors_perArray):

    print('Warning: indices in the arrayNumbers start on 1, not 0. Correcting for this')
    
    arrayList = []
    anchorIndices = []
    
    selectedChannelsIndices = []
    relativeAnchorIdxList = []
    
    availableArrays = np.unique(arrayNumber)
    
    for i in range(availableArrays.shape[0]):
        
        idx = availableArrays[i]

        # print('Finding array ', idx)
        indices = np.where(arrayNumber == idx)[0].astype('int')
        
        array = raw[indices]
        
        quantiles_indices = np.arange(0,1,1.0/(num_anchors_perArray+1))
        quantiles_indices = quantiles_indices[1:] #not using the first one

        # HERE WE SELECT THE ANCHOR POINTS USING QUANTILES OF THE POSSIBLE
        # CHANNELS PER EACH ARRAY. For example, if we have [1,2,3,4,5,6,77,88,99,1002]
        # and 4 anchors, we get 1,3,5,88
        
        # This list contain the anchor indexes but relative to the start of that array.
        # e.g. in [664,788,999,1000,1221] the anchor index per arrayof 788 is 1
        relativeAnchorIdx = []
        
        # print('indices ', indices)
        for q in range(num_anchors_perArray):
            
            quantile = np.quantile(indices, quantiles_indices[q])

            # the next is a way to find the nearest value to the quantile contained
            # in the indices vector
            anchor_index = indices.flat[np.abs(indices - quantile).argmin()]
            
            anchorIndices.append(anchor_index)
            anchor_index_perArray = np.where(indices == anchor_index)[0]
            relativeAnchorIdx.append(anchor_index_perArray)
        
        relativeAnchorIdxList.append(np.array(relativeAnchorIdx))
        arrayList.append(array)
        selectedChannelsIndices.append(indices)

    return arrayList, anchorIndices, relativeAnchorIdxList, selectedChannelsIndices


def create_array_index_new(raw,colsL,num_anchors_perArray, nonvalidE): 
    """
    adapted for the loading3sessions3monkeysData script
    """

    print(raw.shape)

    colorSingleArray = []
    goodChannelsIndex = []
    
    # arrayNumber = np.zeros((raw.shape[0],))
    arrayNumber = np.zeros((1024,))
    # colorsNew = np.zeros((raw.shape[0],3))
    colorsNew = np.zeros((1024,3))

    # for monkey_L data
    counter = 0
    
    #repeatinag color for each array
    for i in range(16):
    
        colorSingleArray.append(colsL[i])
        
        for j in range(64):
            #arrayIndex[counter] = i
            #Same colors as Xings RFs
            arrayNumber[counter] = i + 1
            colorsNew[counter] = colsL[i]
            
            counter = counter + 1   
            
    colorSingleArray = np.array(colorSingleArray)
    
    # #ommiting arrays 2 and 3, which might be located outside V1   
    # if nonvalidE is None:

    #     rawV1 = raw
    # else:
    arrayNumberV1 = np.delete(arrayNumber, nonvalidE, 0)
    colorsNew = np.delete(colorsNew, nonvalidE, 0)
    #     rawV1 = np.delete(raw, nonvalidE, 0)

    #colorSingleArray = np.delete(colorSingleArray, [1,2], axis = 0)
    allIdx = np.linspace(0,1023,1024).astype('int64')
    originalIdx = np.delete(allIdx, nonvalidE)
    
    raw_arrays, anchorIndices, relativeAnchorIdxList, selectedChannelsIndices = separate_arrays_fromIndex(raw,arrayNumberV1,num_anchors_perArray)
    # colsL = arrayIndex
    
    return raw, raw_arrays, anchorIndices, colsL, colorsNew, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, originalIdx


def load_lfp_chunks(chunkyPathRoot = r'LFP/eyes_closed_data/',
                    chunkyPathName = 'LFP_eyes_closed_instance_09082017_resting_state_1024e_chunk_',
                    chunksList = [0], normalize = False):
    """loads a chunk of LFP, corresponding to an entire recording
    we have chunks 0 to 34"""
    
    print('warning: careful, the chunk index should be a 1-index e.g. Matlab index')
    lfpList = None
    
    cont = 0
    if normalize:
        print('normalizing LFP chunks')
    for i in chunksList:
        p = chunkyPathRoot
        
        CHUNKK = i  # NOTATION STARTING FROM 1
        
        print('Loading LFP, 1-indexed')
        print('Loading LFP, CHUNK ', str(CHUNKK))

        name = p + chunkyPathName + str(CHUNKK) + '.npy'
        lfp = np.load(name).astype('float32')
        
        if normalize:
            me = np.expand_dims(lfp.mean(1),1)
            st = np.expand_dims(lfp.std(1),1) 
            lfp =  ( lfp - me ) / st
    
        if cont == 0:
            lfpList = lfp
            
        else:
            lfpList = np.hstack((lfpList,lfp))
    
        cont += 1
        
    return lfpList

def load_lfp_chunks_ALL(chunkyPathRoot = r'LFP/eyes_closed_data/',
                    chunkyPathName = 'LFP_eyes_closed_instance_09082017_resting_state_1024e_chunk_',
                    chunksList = [0], normalize = False):
    """loads a chunk of LFP, corresponding to an entire recording
    we have chunks 0 to 34"""
    
    print('WARNING: careful, the chunk index should be a 1-index e.g. Matlab index')
    print('Loading up to 150 LFP chunks, HARDCODED')
    lfpList = None
    
    cont = 0
    if normalize:
        print('normalizing LFP chunks')

    for i in range(1,151):
        
        try:
            p = chunkyPathRoot
            
            CHUNKK = i#NOTATION STARTING FROM 1
            
            print('Loading LFP,CHUNK, 1-indexed ', str(CHUNKK))
            name = p + chunkyPathName + str(CHUNKK) + '.npy'
            lfp = np.load(name).astype('float32')
            if normalize:
                me = np.expand_dims(lfp.mean(1),1)
                st = np.expand_dims(lfp.std(1),1) 
                lfp =  ( lfp - me ) / st
            if cont == 0:
                lfpList = lfp
                
            else:
                lfpList = np.hstack((lfpList,lfp))
                
        except:
            None
    
        cont += 1
        
    return lfpList


def load_lfp_chunks_ALL_ignoreSecs(chunkyPathRoot = r'eyes_closed_data/',
                    chunkyPathName = 'LFP_eyes_closed_instance_09082017_resting_state_1024e_chunk_',
                    chunksList = [0], normalize = False,
                    ignoreSeconds = 0, samplingFreq = 0):
    
    """loads a chunk of LFP, corresponding to an entire recording
    we have chunks 0 to 34"""
    
    print('WARNING: careful, the chunk index should be a 1-index e.g. Matlab index')
    print('Loading up to 150 LFP chunks, HARDCODED')
    
    total_ignored = 0

    
    lfpList = None
    
    cont = 0
    if normalize:
        print('normalizing LFP chunks')

    for i in range(1, 151):
        
        try:
            p = chunkyPathRoot
            
            CHUNKK = i  # NOTATION STARTING FROM 1
            
            name = p + chunkyPathName + str(CHUNKK) + '.npy'
            lfp = np.load(name).astype('float32')

            if ignoreSeconds > 0:
                
                ignorePoints = ignoreSeconds * samplingFreq
                lfp = lfp[:,ignorePoints:]

            if normalize:
                me = np.expand_dims(lfp.mean(1),1)
                st = np.expand_dims(lfp.std(1),1) 
                lfp =  ( lfp - me ) / st 
                
            if lfp.shape[1] > 0:
                
                if cont == 0:
                    lfpList = lfp

                else:
                    lfpList = np.hstack((lfpList,lfp))
                    
                total_ignored = total_ignored + ignorePoints
                cont += 1  
                
        
            # print('Loaded LFP,CHUNK, 1-indexed ', str(CHUNKK))
  
        except:
            None
    
    return lfpList



def load_lfp_chunks_ALL_ignoreSecs_frequency(chunkyPathRoot = r'LFP/eyes_closed_data/',
                    chunkyPathName = 'LFP_eyes_closed_instance_09082017_resting_state_1024e_chunk_',
                    chunksList = [0], normalize = False,
                    ignoreSeconds = 0, samplingFreq = 0,freqName = '', maxChunkNumber = 150):
    
    """loads and concatenates LFP or MUA chunks, corresponding to an entire recording
       ignores first 4 seconds of data, to avoid visually contaminated data after closing eyes
       
       the difference with this allFrequencies version of the function is just that it stitches and saves
       all frequency chunks (e.g. low, alpha, beta...) and not the original LFP chunks. (it is a trivial difference)
    
    """
    
    # print('WARNING: careful, the chunk index should be a 1-index e.g. Matlab index')
    # print('Loading up to 150 LFP chunks, HARDCODED')
 
    total_ignored = 0
    lfpList = None
    
    cont = 0
    if normalize:
        print('normalizing LFP chunks')
        
    for i in range(1 , maxChunkNumber + 1):
        
        try:
            p = chunkyPathRoot
            
            CHUNKK = i # Notation starts from 1 (Matlab notation)

            name = p + chunkyPathName + str(CHUNKK) + '_' + freqName + '.npy'
            # print(name)
            
            lfp = np.load(name).astype('float32')

            if ignoreSeconds > 0:
                
                ignorePoints = ignoreSeconds * samplingFreq
                lfp = lfp[:,ignorePoints:]

            if normalize:
                me = np.expand_dims(lfp.mean(1),1)
                st = np.expand_dims(lfp.std(1),1) 
                lfp =  ( lfp - me ) / st 
                
            if lfp.shape[1] > 0:
                
                if cont == 0:
                    lfpList = lfp
  
                else:
                    lfpList = np.hstack((lfpList,lfp))
                    
                total_ignored = total_ignored + ignorePoints
                cont += 1  
                
            # print('Loaded LFP,CHUNK, 1-indexed ', str(CHUNKK))
  
        except:
            None
    
    return lfpList

def mike_LFP(signal):
    print('todo lfp')
    
    signal=signal*4
    #fourier1=np.fft.fft(signal)
    #bandpass 500-9000
    fs=30000
    fn=fs/2
    N=len(signal)
    
    b, a = scipy.signal.butter(2, 150/fn, 'lowpass')
    low = scipy.signal.filtfilt(b, a, signal)

    sampling_factor=60
    down=low[:,::sampling_factor]
    
    fs=fs/sampling_factor
    fn=fs/2
    
    for x in [50,100,150]:
        b, a = scipy.signal.butter(2,[(x-2)/fn,(x+2)/fn] , 'bandstop')
        down = scipy.signal.filtfilt(b, a, down)
            
    return down

from scipy.signal import butter, iirnotch, filtfilt, lfilter
import numpy as np

def notch_filter(NOTCH, SAMPLE_RATE,Q = 2.0): # f0 50Hz, 60 Hz
    Q = 2.0  # Quality factor
    # Design notch filter
    b0, a0 = iirnotch(NOTCH , Q, SAMPLE_RATE)
    return b0,a0

def notch_filter_mikel(data, NOTCH, SAMPLE_RATE,Q = 2.0):
    
    b,a = notch_filter(NOTCH, SAMPLE_RATE,Q)
    filtered_data = lfilter(b, a, data)
    
    return filtered_data

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass_filter_filtfilt(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data,axis=-1)
    return y

def calculate_hilbert(data,PLOT=False):
    # Hilbert transform
    
    power = []
    envelope = []

    for i in range(data.shape[0]):
        
        lfpChanD = scipy.signal.detrend(data[i] )

        analytical_signal = hilbert(lfpChanD)
        
        amplitude_envelope = np.abs(analytical_signal)
        hilbert_power = np.abs(analytical_signal)**2
        
        power.append(hilbert_power)
        envelope.append(amplitude_envelope)
        
        if PLOT:
            plt.figure(figsize=fSize)
            plt.plot(analytical_signal.real,linewidth=4, color='b')
            plt.plot(analytical_signal.imag,linewidth=4, color='orange')
            plt.plot(amplitude_envelope,linewidth=4, color='black')
            plt.legend(['Real component','Imaginary component','Envelope'])
            plt.title('Hilbert transform')
            
            plt.figure(figsize=fSize)
            plt.plot(hilbert_power,linewidth=4, color='black')
            plt.legend(['Real component','Imaginary component','Power'])
            plt.title('Hilbert transform power')

    return np.asarray(envelope).astype('float32'), np.asarray(power).astype('float32')

def bandpass_data(data, matt, fs = 500, ORDER=6, PLOT=False, fSize = (10,10)):
    """
    matt == frequency range i.e. [8.0, 13.0]
    
    mattLow       = [  0.5,   8.0]
    mattAlpha     = [  8.0,  13.0]
    mattBeta      = [ 13.0,  30.0]
    mattGamma     = [ 30.0,  80.0]
    mattHighHamma = [ 80.0, 150.0]
    
    """
    
    lowcut = matt[0]
    highcut = matt[1]
    
    filteredList = []
    
    for i in range(data.shape[0]):
        
  
        t1 = time.time()
        x = data[i]
        t = np.arange(0,x.shape[0])
        T = 1.0 / fs
        t = np.linspace(0.0, 1.0/(T), x.shape[0]//1)
        y = butter_bandpass_filter_filtfilt(x, lowcut, highcut, fs, order=ORDER)
        t2 = time.time()
        
        if PLOT:
            plt.figure(figsize=fSize)
            
            plt.plot(np.arange(x.shape[0]), y, 
                     label='Filtered signal ' + str(matt[0]) + ' ' + str(matt[1]),
                     color = 'black', linewidth = 3)    
            plt.xlabel('data point')
            plt.title('Bandpass Filtered data')
            plt.axis('tight')
            plt.legend(loc='upper left')
            
            plt.show()
            
        filteredList.append(y.astype('float32'))

    return np.asarray(filteredList,dtype='float32')#.astype('float32')


def create_array_index(raw,colsL,num_anchors_perArray, nonvalidE):
    
    colorSingleArray = []
    goodChannelsIndex = []
    
    arrayNumber = np.zeros((raw.shape[0],))
    arrayIndex = np.zeros((raw.shape[0],3))
    
    # for monkey_L data
    counter = 0
    
    for i in range(16):
    
        colorSingleArray.append(colsL[i])
        
        for j in range(64):
            #arrayIndex[counter] = i
            #Same colors as Xings RFs
            arrayNumber[counter] = i + 1
            arrayIndex[counter] = colsL[i]
            
            counter = counter + 1   
            
    colorSingleArray = np.array(colorSingleArray)
    
    #ommiting arrays 2 and 3, which might be located outside V1   
    if nonvalidE is None:
        arrayNumberV1 = arrayNumber
        arrayIndexV1 = arrayIndex
        rawV1 = raw
    else:
        arrayNumberV1 = np.delete(arrayNumber, nonvalidE, 0)
        arrayIndexV1 = np.delete(arrayIndex, nonvalidE, 0)
        rawV1 = np.delete(raw, nonvalidE, 0)
    
    #colorSingleArray = np.delete(colorSingleArray, [1,2], axis = 0)
    allIdx = np.linspace(0,1023,1024).astype('int64')
    originalIdx = np.delete(allIdx, nonvalidE)
    
    raw_arrays, anchorIndices, relativeAnchorIdxList, selectedChannelsIndices = np.array(separate_arrays_fromIndex(rawV1,arrayNumberV1,num_anchors_perArray))
    colsL = arrayIndexV1
    
    return rawV1, raw_arrays, anchorIndices, colsL, arrayIndexV1, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, originalIdx

def load_utah_xy(wherePath,matName):
    
    # a = loadmat(path+matName)
    a = loadmat(os.path.join(wherePath, matName))

    x_coords = a['allMiddleXCoordsGrid']
    y_coords = a['allMiddleYCoordsGrid']
    
    return x_coords, -y_coords + abs(-y_coords.max())


def load_rf_instances(path_root):
    #print(f"DEBUG: load_rf_instances called with path_root: {path_root}")
    rf_list = []
    for i in range(8): 
        try:
            # Try new method first (os.path.join)
            rf_file = os.path.join(path_root, f'RFs_instance{i+1}.mat')
            #print(f"DEBUG: Trying to load: {rf_file}")
            rf1 = loadmat(rf_file)
            rfs = rf1['RFs']
            channelRFs = rf1['channelRFs'] # Units are pixels (here its 26 pixels per degree of visual angle (dva))
            rf_locations = channelRFs[:,0:2]
            rf_list.append(rf_locations)
            #print(f"DEBUG: Successfully loaded RF instance {i+1}")
        except Exception as e:
            #print(f"DEBUG: First method failed for instance {i+1}: {e}")
            try:
                # Fallback to old method for backward compatibility
                old_path = path_root + 'RFs_instance' + str(i+1) + '.mat'
                #print(f"DEBUG: Trying fallback: {old_path}")
                rf1 = loadmat(old_path)
                rfs = rf1['RFs']
                channelRFs = rf1['channelRFs'] # Units are pixels (here its 26 pixels per degree of visual angle (dva))
                rf_locations = channelRFs[:,0:2]
                rf_list.append(rf_locations)
                #print(f"DEBUG: Successfully loaded RF instance {i+1} with fallback")
            except Exception as e2:
                print(f'Error loading RF instance {i+1}: {e2}')

    #print(f"DEBUG: rf_list length: {len(rf_list)}")
    return(np.asarray(rf_list,dtype='float32'))


def order_RF_instances(rf_list, arrayNums, channelNums, V1ONLY=False):

    if V1ONLY:
        toTake = np.hstack((np.linspace(0,63,64), np.linspace(192,1023,1024-192))).astype(int)
    else:
        toTake =np.linspace(0,1023,1024).astype(int)
    
    orderedRfs = None
    rfArrs = []
    
    # for each of the 8 instances, order the arrays
    for insC in range(rf_list.shape[0]):
        numeritos = arrayNums[:,insC]
        
        # find which arrays are in each instance
        whichArr = np.unique(numeritos)
        
        # for each of the arrays, get the data and order it
        for w in whichArr:
            # where are the arrays channels
            donde = np.where(numeritos == w)
            
            # getting rfs for that array
            rfArr = rf_list[insC][donde]
            
            # what is their real index of those arrays
            canalesInd = channelNums[:,insC][donde] - 1
            rfArr = rfArr[canalesInd]
                    
            if orderedRfs is None: 
                orderedRfs = rfArr
    
            else:
                orderedRfs = np.vstack((orderedRfs,rfArr))
                
            rfArrs.append(rfArr)
            
    rfArrs     = np.asarray(rfArrs,dtype='float32')    
    orderedRfs = np.asarray(orderedRfs,dtype='float32')    
 
    return rfArrs, orderedRfs


def order_LFP_instances(lfp_list,arrayNums, channelNums, V1ONLY=False):
    
    if V1ONLY:
        toTake = np.hstack((np.linspace(0,63,64), np.linspace(192,1023,1024-192))).astype(int)
    else:
        toTake =np.linspace(0,1023,1024).astype(int)
        
    a_lfp = np.asarray(lfp_list)
    chunkList = []
    MEANCHUNKY = []
    
    for c in range(a_lfp.shape[1]):
        chunk = a_lfp[:,c]
        orderedChannels = None
        meanchunky = None
        chunked = None
    
        # for each of the 8 instances, order the arrays
        for insC in range(chunk.shape[0]):  
            numeritos = arrayNums[:,insC]
            
            # find which arrays are in each instance
            whichArr = np.unique(numeritos)
            
            # for each of the arrays, get the data and order it
            for w in whichArr:
                # where are the arrays channels
                donde = np.where(numeritos == w)
                
                # getting channels for that array
                chArr = chunk[insC][donde]
                
                # what is their real index of those arrays
                #print('WARNING, CHECK THIS ORDERING. It could be that first you need to take the channel and order it afterwards')
                # canalesInd = electrodeNums[:,insC][donde] - 1
                canalesInd = channelNums[:,insC][donde] - 1
                chArr = chArr[canalesInd]
                meansita = np.mean(chArr,0)
                
                if orderedChannels is None:
                    
                    orderedChannels = chArr
                    meanchunky = meansita
                else:
                    orderedChannels = np.vstack((orderedChannels,chArr))
                    meanchunky = np.vstack((meanchunky, meansita))
                    
        if chunked is None:
            chunked = orderedChannels[toTake]
        else:
            chunked = np.vstack(orderedChannels[toTake])
            
            
        MEANCHUNKY.append(meanchunky)
        chunkList.append(chunked)
            
        # else:
        #     print('Empty data chunk ')
            
    return chunkList, MEANCHUNKY


def load_lfp_instances_OpenEyes_frequencies(folder_path, freq_bands):
    
    """
    Load LFP data for multiple instances and filter it into different frequency bands.
    
    (data is Z - scored)

    Parameters:
    folder_path (str): The path to the folder containing the LFP instances.
    freq_bands (dict): Dictionary with keys as frequency band names and values as 
                       tuple of (low_freq, high_freq) for band-pass filter.

    Returns:
    dict: A dictionary with frequency band names as keys and lists of 2D numpy arrays
          (channels x samples) as values.

    Example usage:
    freq_bands = {'delta': (0.1, 4), 'theta': (4, 8), 'alpha': (8, 14)}
    lfp_data = load_lfp_instances_OpenEyes_frequencies('./data/', freq_bands)
    """
    
    freq_sampling = 500  # Sample frequency in Hz
    seconds = 0.3
    dataInterval = int(seconds * freq_sampling)
    
    lfp_list = {}
    lfp_list['original'] = []
    
    # Initialize lists for each frequency band
    for band in freq_bands.keys():
        lfp_list[band] = []
    
    for i in range(8):  # Example range, adjust according to your data
        try:
            ins = scipy.io.loadmat(folder_path + 'LFP_instance' + str(i+1) + '.mat')['channelDataLFP'][0]
            print(f"Loaded LFP_instance {i+1}")
            
            chan_dataList = {}
            chan_dataList['original'] = []
            
            # Initialize lists for each frequency band
            for band in freq_bands.keys():
                chan_dataList[band] = []
            
            for channel in range(128):  # Example range, adjust according to your data
                chan_data = ins[channel]

                
                # Filter LFPs for different frequency bands and store them
                for band, freq_range in freq_bands.items():
                    
                    filtered_data = butter_bandpass_filter(chan_data, freq_range[0], freq_range[1], freq_sampling)

                    # Normalize the filtered data
                    mean = np.mean(filtered_data)
                    std = np.std(filtered_data)
                    normalized_data = (filtered_data - mean) / std
                    chan_dataList[band].append(np.hstack(normalized_data[:, :dataInterval]))

                    # chan_dataList[band].append(np.hstack(filtered_data[:, :dataInterval]))
                    
                    
                # Normalize the original LFP data
                mean_original = np.mean(chan_data)
                std_original = np.std(chan_data)
                normalized_original = (chan_data - mean_original) / std_original
                chan_dataList['original'].append(np.hstack(normalized_original[:, :dataInterval]))

                # Store original LFP
                # chan_dataList['original'].append(np.hstack(chan_data[:, :dataInterval]))

            # Store the LFPs for each frequency band
            for band in chan_dataList.keys():
                lfp_list[band].append(np.array(chan_dataList[band]))
            
            print(f"LFP_instance {i+1} processed")
            
        except Exception as e:
            print(f'Error loading lfp instance {i+1}: {e}')

    print("All LFP instances processed")
    return lfp_list

def load_lfp_instances_monkey_A(folder_path):
    
    """
    
    monkey_A version of load lfp instances. Instances are called 'LFP_eyes_closed_NSPX'

    lfp_list will contain all the 8 instances of LFP. Each instance contains a list of chunks
    of data where the eyes were closed
    """
    lfp_list = []
    
    for i in range(8):
        
        try:
            lfp_list.append(scipy.io.loadmat(folder_path + 'LFP_eyes_closed_NSP' + str(i+1) + '.mat')['LFP_eyes_closed_data'][0])

        except:
            print('Error loading lfp instance ', i+1)
        
    return lfp_list


def load_lfp_instances_monkey_L(folder_path):
    
    """
    monkey_L version of load lfp instances. Instances are called 'LFP_eyes_closed_instanceX'
    lfp_list will contain all the 8 instances of LFP. Each instance contains a list of chunks
    of data where the eyes were closed
    """
    lfp_list = []
    
    for i in range(8):
        
        try:
            lfp_list.append(scipy.io.loadmat(folder_path + 'LFP_eyes_closed_instance' + str(i+1) + '.mat')['LFP_eyes_closed_data'][0])
        except:
            print('Error loading lfp instance ', i+1)
        
    return lfp_list

def load_lfp_instances(folder_path):
    
    """
    This is equal to the monkey_L version
    
    lfp_list will contain all the 8 instances of LFP. Each instance contains a list of chunks
    of data where the eyes were closed
    """
    lfp_list = []
    
    for i in range(8):
        
        try:
            print(folder_path + 'LFP_eyes_closed_instance' + str(i+1) + '.mat')
            lfp_list.append(scipy.io.loadmat(folder_path + 'LFP_eyes_closed_instance' + str(i+1) + '.mat')['LFP_eyes_closed_data'][0])
        except:
            print('Error loading lfp instance ', i+1)
        
    return lfp_list

def load_lfp_instances_OpenEyes(folder_path):
    
    """
    
    monkey_A version of load lfp instances but for Open Eyes data. 
    Instances are called 'LFP_instance_NSPX'

    lfp_list will contain all the 8 instances of LFP. Each instance contains a list of chunks
    of data where the eyes were closed
    """
    lfp_list = []
    
    for i in range(8):
        
        try:
            #lfp_list.append(scipy.io.loadmat(folder_path + 'LFP_eyes_closed_NSP' + str(i+1) + '.mat')['LFP_eyes_closed_data'][0])
            # ins = scipy.io.loadmat(folder_path + 'LFP_instance' + str(i+1) + '.mat')['LFP_eyes_closed_data'][0]
            ins = scipy.io.loadmat(folder_path + 'LFP_instance' + str(i+1) + '.mat')['channelDataLFP'][0]
    
            #############################
            seconds = 0.3
            freq_sampling = 500
            # calculating the amount of data samples to take
            # to get only spontaneous data
            
            dataInterval = int( seconds * freq_sampling )
            chan_dataList = []
            for channel in range(128):
            
            
                # going through each channel's data 
                # (each data containing many trials)
            
                chan_data = ins[channel]
                
                # using only the first 300 ms of data
                chan_data = chan_data[:, 0 : dataInterval]
                
                # appending data
                chan_dataList.append(np.hstack(chan_data))
            
            ins = np.asarray(chan_dataList)
            ###############################
            
            lfp_list.append(ins)
        

        except:
            print('Error loading lfp instance ', i+1)
        
    return lfp_list

def euclidean_Ds(envList, PLOT = False):
    
    """
    Calculates euclidean distance matrices for each element of the list.
    Each elements corresponds to a whole bunch of channels
    """
    
    if len(envList.shape) != 3:
        envList = np.expand_dims(envList,0)
        # print('adding 1 axis to list of channels')
    else:
        None
        
    Dlist = []
    for i in range(envList.shape[0]):
        D = euclidean_distances( envList[i] )
        Dlist.append(D)
        if PLOT:
            plt.figure();plt.imshow(D,cmap='afmhot');plt.colorbar()
    return np.array(Dlist)


def corr_Ds(envList, PLOT = False):
    
    """
    Calculates correlation matrices for each element of the list.
    Each elements corresponds to a whole bunch of channels
    """

    if len(envList.shape) != 3:
        envList = np.expand_dims(envList,0)
        # print('adding 1 axis to list of channels')
    else:
        None
        
    Dlist = []         
    for i in range(envList.shape[0]):
        D = np.corrcoef( envList[i] ) #, dtype = 'float32' )
        Dlist.append(D)
        if PLOT:
            plt.figure();plt.imshow(D,cmap='afmhot');plt.colorbar()
    return np.array(Dlist)


def calculate_corr_of_distances(mds,utah):
    """gets a bunch of 2d points, calculates distance matrices
    and calculates the correlations between those distance matrices"""
    D = euclidean_distances( mds)
    U = euclidean_distances( utah)
    cm = np.corrcoef(D.flat,U.flat)
    return cm[0,1]

def load_utah_XING(wherePath,matName,allColors,PLOT=False):
    
    """
    Loads utah coordinates with the right order,
    it also plots them if commanded
    """
    counterColors = 0
    utahGood = []
    
    x_utah, y_utah = load_utah_xy(wherePath,matName)
    
    if PLOT: plt.figure(figsize=(15,10))
    for i in range(16):
        counter = 0
        x,y = x_utah[:,:,i], y_utah[:,:,i]
    
        for j in range(x.shape[0]):
            for k in reversed(range(x.shape[1])):  
                if PLOT:
                    plt.scatter(x[j,k], y[j,k], color = allColors[counterColors], s = 100)
                    plt.text(x[j,k] * (1 + 0.01), y[j,k] * (1 + 0.01) , 
                             str(counter + 1), fontsize=8)
                counter += 1
                counterColors   += 1
                utahGood.append(np.array([x[j,k],y[j,k]]))
    utahGood = np.asarray(utahGood, dtype = 'float32')
    return utahGood

def multiple_bandpass(data, whichBands, freQsampling, Order = 3, Plot = False, fSize = (10,10)):
    
    """
    
    - perform bandpass filter in different freq bands
    
    - whichBands looks like :
    
    array([[  0.5,   8. ],
       [  8. ,  13. ],
       [ 13. ,  30. ],
       [ 30. ,  80. ],
       [ 80. , 150. ]])
    
    the whichBands array shape in this example is (5,2)
    
    """
    
    # print('applying bandpass filter to the desired freq bands')
    num_bands = whichBands.shape[0]
    bandpassedList = []
    
    for i in range(num_bands):
        band = whichBands[i]
        bandpassedList.append( bandpass_data(data, band, fs = freQsampling, 
                                            ORDER=Order, PLOT=Plot, fSize = (10,10)) )
    return np.array(bandpassedList)


def scipy_Antonio_procrustes(data1, data2):
    r"""
    
    Antonio: I just modified this to return also R and s
    
    Procrustes analysis, a similarity test for two data sets.
    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:
    - :math:`tr(AA^{T}) = 1`.
    - Both sets of points are centered around the origin.
    Procrustes ([1]_, [2]_) then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
    pointwise differences between the two input datasets.
    This function was not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), simply add columns of zeros to the smaller
    of the two.
    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.
    Raises
    ------
    ValueError
        If the input arrays are not two-dimensional.
        If the shape of the input arrays is different.
        If the input arrays have zero columns or zero rows.
    See Also
    --------
    scipy.linalg.orthogonal_procrustes
    scipy.spatial.distance.directed_hausdorff : Another similarity test
      for two data sets
    Notes
    -----
    - The disparity should not depend on the order of the input matrices, but
      the output matrices will, as only the first output matrix is guaranteed
      to be scaled such that :math:`tr(AA^{T}) = 1`.
    - Duplicate data points are generally ok, duplicating a data point will
      increase its effect on the procrustes fit.
    - The disparity scales as the number of points per input matrix.
    References
    ----------
    .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
    .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".
    Examples
    --------
    >>> from scipy.spatial import procrustes
    The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
    ``a`` here:
    >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
    >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
    >>> mtx1, mtx2, disparity = procrustes(a, b)
    >>> round(disparity)
    0.0
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    
    mean1 = np.mean(mtx1, 0)
    mean2 = np.mean(mtx2, 0)
    
    mtx1 -= mean1
    mtx2 -= mean2
    
    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R, s, norm1, norm2, mean1, mean2


def rescale_procrustes_map(H, R, s, norm1, norm2, mean1, mean2):
    
    """
    Input: H (recovered map) + procrustes parameter results
    Output: rescaled and aligned recoveredH map
    
    After using procrustes analysis to align the "utah" map with the
    "H" (recovered) map, rescales the H map so it matches the original utah 
    scale, this is, returns an H map that is aligned with the original utah map
    on  same original utah map scale
    
    note: norm1, mean1 are the parameters related to the utah array or anchor points,
    norm2, mean2 are the ones for the recovered H map
    """
    
    proH = ( np.dot( ((H-mean2) / norm2), R.T ) * s )
    recoveredH =  (proH * norm1 ) + mean1 
    
    return recoveredH

########### THESE FUNCTIONS ARE FOR THE REGRESSION ANALYSIS

def get_triangleList(corrDlist,utah_distances, pixels_per_mm = 14.0):
    
    """
    WARNING: here  the input  utah_distances is already normalized between 0 and 1,
    thus we multiply by pixels_per_mm
    Input: corrDlist is a list of correlation matrix
           utah_distances
    Output: the upper triangular distance matrix for the correlation matrices list 
    and for the utah distances matrix
    """
    
    # indexes of the upper triangular distance matrix, to avoid repeated numbers
    a,b = np.triu_indices(utah_distances.shape[0])[0], np.triu_indices(utah_distances.shape[0])[1]
    # indexes of the upper triangular distance matrix, only calculated once to avoid repeated numbers
    utahTriangle = utah_distances[a,b] * pixels_per_mm #14 is pixels per mm
    
    # selecting all the distances to plot
    dTriangleList = []

    cont = 0
    for i in range(len(corrDlist)):
            #D = euDlist[cont][0,:,:]
            D = corrDlist[i][0,:,:]
            dTriangleList.append( D[a,b] )

    return utahTriangle, np.asarray(dTriangleList,'float32')


# Calculating means and standard deviations

def mean_and_std_dTriangleList(dTriangleList, utahTriangle, step = 0.1, PLOT=False, s = 0.05, nameList = ['lfp','low', 'alpha', 'beta', 'gamma', 'high_gamma', 'MUA']):
    
    """
    returns the binned correlation vs distance arrays
    """

    # nameList = ['low', 'alpha', 'beta', 'gamma', 'high_gamma', 'MUA1', 'MUA10', 'MUA100']
    
    
    # Calculating means and standard deviations
    uStdList    = []
    uMeanList   = []
    uCountList  = []
    
    dSortedList = []
    
    for DE in range(len(dTriangleList)):
        
        dTriangle = dTriangleList[DE]
        
        # finding sorting indices for cortical distances
        # to order correlations between electrodes by distance from the electrodes
        uSort = np.argsort(utahTriangle)
        uSorted = utahTriangle[uSort]
        dSorted = dTriangle[uSort]
        
        # Binning data
        step = step
        
        # bins goes from 0 to the max cortical distance
        bins = np.arange(0,int(np.ceil(dSorted.max())),step)
    
        uBinsMean = []
        uBinsStd = []
        uBinsCount = []
        
        dBins = []
        
        for i in range(bins.shape[0]):
            
            #finding range limits
            idx1 = np.where(dSorted>=bins[i])
            idx2 = np.where(dSorted<bins[i]+step)
            idx = np.intersect1d(idx1, idx2)
    
            #appending data within those range limits
            dBins.append(bins[i])
            uBinsMean.append(np.mean(uSorted[idx]))
            uBinsStd.append(np.std(uSorted[idx]))
            uBinsCount.append(idx.shape[0])
            
        uBinsMean = np.asarray(uBinsMean,'float32')
        uBinsStd = np.asarray(uBinsStd,'float32')
        uBinsCount = np.asarray(uBinsCount,'float32')
    
        uStdList  .append(uBinsStd)
        uMeanList .append(uBinsMean)
        uCountList.append(uBinsCount)
    
        # original
        # if PLOT:
        #     plt.figure(dpi = 180)  
        
        #     plt.scatter(dSorted,uSorted, s = s, color = 'red') 
        #     plt.plot(dBins, uBinsMean, 'k-')
        #     plt.fill_between(dBins,uBinsMean-uBinsStd,uBinsMean+uBinsStd,
        #                       facecolor="yellow",
        #                       alpha=0.4)
            
        if PLOT:
            plt.figure(dpi = 180)  
        
            plt.scatter(uSorted, dSorted, s = s, color = 'red') 
            plt.plot(dBins, uBinsMean, 'k-')
            plt.fill_between(dBins,uBinsMean-uBinsStd,uBinsMean+uBinsStd,
                              facecolor="yellow",
                              alpha=0.4)
            
            plt.title('cortical distance mms ' + nameList[DE], size = 20)
            plt.xlabel('Correlation')#, size = 20)
            plt.xlabel('Cortical distance')#, size = 15)

            plt.xlim(-0.5,1)
            # plt.ylim(-1,33)
            plt.show()
            
        dSortedList.append(dSorted)
    
    dBins = np.asarray(dBins,dtype='float32')
    uStdList = np.asarray(uStdList,dtype='float32')
    uMeanList = np.asarray(uMeanList,dtype='float32')
    uCountList = np.asarray(uCountList,dtype='float32')
    dSortedList = np.asarray(dSortedList,dtype='float32')
    
    return uMeanList, uStdList, uCountList, dSortedList, uSorted, dBins

def plot_hex_cortVscorr( dTriangleList, utahTriangle, utah_distances, arrayIndexV1, 
                        gridsize=300, vmax = 30, 
                        nameList = ['lfp','low', 'alpha', 'beta', 'gamma', 'high_gamma', 'MUA'],
                        dpi = 300, SAVEFIG = 'False', savePath = '', subject = ' '):
    
    
    """
    Plot hexbin plots showing correlation vs cortical distance for different frequency bands and MUA.
    
    Parameters:
    dTriangleList (list of np.ndarray): List of upper-triangular correlation matrices.
    utahTriangle (np.ndarray): Upper-triangular matrix of Utah array distances.
    utah_distances (np.ndarray): Matrix of Utah array distances.
    arrayIndexV1 (np.ndarray): Color codes to represent different areas in the Utah array.
    gridsize (int): Size of the hexbin grid. Default is 300.
    vmax (int): Maximum count for hexbin color coding. Default is 30.
    nameList (list): Names of the frequency bands and MUA for labeling. Default includes standard bands and MUA.
    dpi (int): DPI for saved figures. Default is 300.
    SAVEFIG (bool): Whether to save the generated figures. Default is False.
    savePath (str): Directory path to save the figures.
    subject (str): Name of the subject for labeling.
    
    Returns:
    None
    """
    
    from sklearn.metrics import r2_score
    
    font = {'family' : 'normal',
            #'weight' : 'bold',
            'size'   : 7}
    
    plt.rc('font', **font)
    
    a,b = np.triu_indices(utah_distances.shape[0])[0], np.triu_indices(utah_distances.shape[0])[1]
    colorcitos = arrayIndexV1[b]
    cont = 0
    
    fig = plt.figure(figsize = (10,6),  dpi = dpi)
    print('dpi ', dpi)
    for i in range(2):
        for j in range(3):
    

                print(cont)
                dTriangle = dTriangleList[cont] #D[a,b] 
                
                print(dTriangle.mean())
                # dTriangle = np.arctanh(dTriangle)
                
                print('WARNING: clipping data')
                dTriangle[dTriangle > 2.5] = 2.5

                ax = plt.subplot2grid((2,3), (i,j))  

                ######### Fitting exponential curve
                        
                from scipy.optimize import curve_fit
                
                def func(x, a, b, c):
                    return a * np.exp(-b * x) + c
        
                x = utahTriangle
                y = dTriangle
        
                popt, pcov = curve_fit(func, x, y)

                newX_forR2 = utahTriangle                
                newX_toPlot = np.linspace(0,30,200)  
                result_forR2 = func(newX_forR2, *popt)  
                result_toPlot = func(newX_toPlot, *popt)
    
                r2 = r2_score(y , result_forR2)
                
                labelita = 'decay: '+ str( np.around(popt[1], decimals=2) )
                labelita = labelita + '\n r2: '+ str( np.around(r2, decimals=2) )

                ax.plot(newX_toPlot, result_toPlot,  color = 'white', ls = '-.', linewidth = 2.0, label = labelita)
                ax.text(utahTriangle.max() - utahTriangle.max()*0.5, .95, labelita, ha='right', va='top',   fontsize = 7, color = 'white')

                ######### Density hex plot

                ax.hexbin(utahTriangle, dTriangle, cmap='seismic',
                           gridsize=gridsize, vmax = vmax, alpha = 1)
                
                ax.set_title(nameList[cont])
                # ax.set_xlabel('(1 / cortical distance) in mms')
                # plt.xticks(np.arange(0,15,2))

                if cont == 0 or cont == 3:
                    ax.set_ylabel('  correlation  ')
                if cont == 3 or cont == 4 or cont == 5:
                    
                    ax.set_xlabel('cortical distance (mm)')
                # ax.set_xlim(0,  5)
                ax.set_ylim(-0.1,  1.1)
                ax.set_xlim(-0.1,  utahTriangle.max())

                cont += 1

    fig.suptitle('Cortical distance vs correlation ', fontsize=15) #, x = 0.27, y = 0.9, fontsize = 20) # or plt.suptitle('Main title')
    fig.tight_layout()

    if SAVEFIG:
        fig.savefig(savePath + subject + '_DistanceVsCorr10-6.tif' , dpi = 500)
        fig.savefig(savePath + subject + '_DistanceVsCorr10-6.pdf',  dpi = 500)
        fig.savefig(savePath + subject + '_DistanceVsCorr10-6.svg' , dpi = 500)

    
def train_regression_multiBand(c_distances, correlations, bands, exponentRange, 
                                ridgeAlpha = 0.1, 
                                poly_degree = 1, NUM_FOLDS = 3,
                                include_bias=True,
                                printBandsName=True):
    bestRstdList = []
    bestRegrList = []
    bestRmeanList = []
    bestExponentList = []
    
    # # For every frequency band
    # for SELECTED in bands:
    print(f'Calculating  {NUM_FOLDS} fold crossvalidation for bands:')
    for name in nameList:
        print(str(name))
    
    expList = []
    rMeanList = []
    rStdList = []
    regrList = []
    
    # Find the right transformation exponent
    # and the crossvalidated prediction accuracy
    for exp in exponentRange:
    # for exp in np.linspace(-1,-0.5,10):
    
        Exponent=exp
      
        # inv_dist=np.power( (1 / utahTriangle) ,Exponent)
        inv_dist=np.power( 1 / c_distances , Exponent)
    
        #% Linear regression between the correlations and true distances
        # selected = np.array(([SELECTED]))
        selected = bands
    
        # Create linear regression object
        regr = Ridge(alpha = ridgeAlpha)
    
        poly_features = PolynomialFeatures(degree=poly_degree, include_bias=include_bias) # Degree M=2 with bias term
        Phi = poly_features.fit_transform(correlations[selected].T)
        
        # # Train the model using the training sets
        cv = KFold(n_splits=NUM_FOLDS, random_state=42, shuffle=True)
        # scores = cross_val_score(regr, correlations[selected].T, inv_dist, 
        #                          cv=cv, scoring='r2', n_jobs=-1)
        scores = cross_val_score(regr, Phi, inv_dist, 
                                  cv=cv, scoring='r2', n_jobs=-1)
        
        print('scores ', np.round(scores,3))
        # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        rMeanList.append(scores.mean())
        rStdList.append(scores.std())
        expList.append(exp)
        regrList.append(regr)
        #print(f'Selected {SELECTED} exponent {Exponent} r score {r2_score(inv_dist, pred)} ')
    
    maxi = np.where(np.array(rMeanList) == np.array(rMeanList).max())[0][0]
    
    # appending the best exponent and results for each frequency band
    bestRmeanList.append(rMeanList[maxi])
    bestRstdList.append(rStdList[maxi])
    bestExponentList.append(expList[maxi])
    
    # fitting the model with all the anchors data before returning it
    inv_dist=np.power( c_distances , expList[maxi])
    regr = regrList[maxi].fit(Phi,inv_dist)
    print('PHI SHAPE ', Phi.shape)
    bestRegrList.append(regr) 

    return bestRegrList,bestRmeanList,bestRstdList,bestExponentList

def predict_distancesMultiband(regrList,c_distancesALL,correlationsALL,corrMatricesALL,poly_degree,
                      include_bias,
                      bestExponentList,nameList):
    predictionList = []
          
    exponent = bestExponentList[0]

    inv_dist = np.power( 1 / c_distancesALL , exponent)
    regr = regrList[0]
    
    # # Make predictions using the testing set
    poly_features = PolynomialFeatures(degree=poly_degree, include_bias=include_bias) # Degree M=2 with bias term
    Phi = poly_features.fit_transform(correlationsALL.T)
    pred = regr.predict(Phi)
    
    # Predicting on the distance matrices
    toPredict = corrMatricesALL.reshape(len(bands),-1).T
    toPredictPhi = poly_features.fit_transform(toPredict)
    
    predicted = regr.predict(toPredictPhi)
    
    predictedMatrix = predicted.reshape(corrMatricesALL[0].shape)
    predictedMatrix = np.power(predictedMatrix,1/-exponent)
    
    plt.figure(dpi = 180)
    plt.imshow(predictedMatrix)
    plt.colorbar()
    predictionList.append(predictedMatrix)
    
    # making figures
    plt.figure(dpi = 180)
    plt.scatter(pred,inv_dist, s = 20, color = 'black')
    plt.title(f'{nameList[i]} regression vs real data', size = 20)
    plt.xlabel('Prediction regression model', size = 20)
    plt.ylabel(f'Inverse distance ^ {e}', size = 20)
    
    return predictionList, pred
   
def train_regression_singleBand(c_distances, correlations, bands, exponentRange, 
                                ridgeAlpha = 0.1, 
                                poly_degree = 1, NUM_FOLDS = 3,
                                include_bias=True,
                                printBandsName=True):
    
    bestRstdList = []
    bestRegrList = []
    bestRmeanList = []
    bestExponentList = []

    # For every frequency band
    for SELECTED in bands:
        
        if printBandsName:
            print(f'Calculating  {NUM_FOLDS} fold crossvalidation for band ' + str(nameList[SELECTED]))
        
        expList = []
        rMeanList = []
        rStdList = []
        regrList = []
    
        # Find the right transformation exponent
        # and the crossvalidated prediction accuracy
        for exp in exponentRange:
        # for exp in np.linspace(-1,-0.5,10):
    
            Exponent=exp
      
            # inv_dist=np.power( (1 / utahTriangle) ,Exponent)
            inv_dist=np.power( 1 / c_distances , Exponent)
    
            #% Linear regression between the correlations and true distances
            selected = np.array(([SELECTED]))
            # Create linear regression object
            regr = Ridge(alpha = ridgeAlpha)
        
            poly_features = PolynomialFeatures(degree=poly_degree, include_bias=include_bias) # Degree M=2 with bias term
            Phi = poly_features.fit_transform(correlations[selected].T)
            
            # # Train the model using the training sets
            cv = KFold(n_splits=NUM_FOLDS, random_state=42, shuffle=True)
            # scores = cross_val_score(regr, correlations[selected].T, inv_dist, 
            #                          cv=cv, scoring='r2', n_jobs=-1)
            scores = cross_val_score(regr, Phi, inv_dist, 
                                      cv=cv, scoring='r2', n_jobs=-1)
            
            # print('scores ', scores)
            # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
            rMeanList.append(scores.mean())
            rStdList.append(scores.std())
            expList.append(exp)
            regrList.append(regr)
            #print(f'Selected {SELECTED} exponent {Exponent} r score {r2_score(inv_dist, pred)} ')
    
        maxi = np.where(np.array(rMeanList) == np.array(rMeanList).max())[0][0]
    
        # appending the best exponent and results for each frequency band
        bestRmeanList.append(rMeanList[maxi])
        bestRstdList.append(rStdList[maxi])
        bestExponentList.append(expList[maxi])
        
        # fitting the model with all the anchors data before returning it
        inv_dist=np.power( 1 / c_distances , expList[maxi])
        regr = regrList[maxi].fit(Phi,inv_dist)
        bestRegrList.append(regr) 
    
    return bestRegrList,bestRmeanList,bestRstdList,bestExponentList

def predict_distances_SingleBand(regrList,c_distances,correlations,corrMatrices,num_bands,poly_degree,
                      include_bias,
                      bestExponentList,nameList):
    predictionList = []
    for i in range(num_bands):
        
        name = nameList[i]
        exponent = bestExponentList[i]
    
        inv_dist = np.power( 1 / c_distances , exponent)
        regr = regrList[i]
    
        # # Make predictions using the testing set
        poly_features = PolynomialFeatures(degree=poly_degree, include_bias=include_bias) # Degree M=2 with bias term
        Phi = poly_features.fit_transform(correlations[[i]].T)
        pred = regr.predict(Phi)
        pred = 1 / np.power(pred,-exponent)

        
    
        # Predicting on the distance matrices
        toPredict = corrMatrices[i].reshape(1,-1).T
        toPredictPhi = poly_features.fit_transform(toPredict)
    
        # transforming back to distance
        predicted = regr.predict(toPredictPhi)
        # predicted = np.power(predicted,1/-exponent)

        # transforming back to distance
        predictedMatrix = predicted.reshape(corrMatrices[0].shape)
        
        
        predictedMatrix = 1 / np.power(predictedMatrix,-exponent)

        #predictedMatrix = np.power(predictedMatrix,1/-exponent)
    
        plt.figure(dpi = 180)
        plt.imshow(predictedMatrix)
        plt.colorbar()
        predictionList.append(predictedMatrix)
        
        # making figures
        plt.figure(dpi = 180)
        plt.scatter(pred,c_distances, s = 20, color = 'black')
        plt.title(f'{nameList[i]} regression vs real data ', size = 20)
        plt.xlabel('Prediction regression model', size = 20)
        plt.ylabel(f'Inverse distance ^ {e}', size = 20)
        
    return predictionList, pred
######################################### END FUNCTIONS REGRESSION ANALYSIS


def calculate_EU_CORR_matrices(lfp, mua, plotSi = False):
    
    """
    calculates distance matrices for the multi band pass LFP and MUA
    and appends them
    """
    
    # Calculate distance matrices for all the frequencies
    corrDlist_LFP = []
    euDlist_LFP = []
    
    
    for i in range(lfp.shape[0]):
        corrDlist_LFP.append(corr_Ds(lfp[i], PLOT = plotSi))
        euDlist_LFP.append(euclidean_Ds(lfp[i], PLOT = plotSi))
        
    # for MUA_INSTANCE in [0]: # I have all MUA instances together
    print('Adding MUA to the correlation matrix list')
    corrDlist_LFP.append(corr_Ds(mua, PLOT = plotSi))
    euDlist_LFP.append(euclidean_Ds(mua, PLOT = plotSi))
    
    return corrDlist_LFP, euDlist_LFP


def load_monkey_data(monkey_LorA,EXCLUDE_V4, CHUNKS, STANDARIZE_LFP, num_anchors_perArray,
                     newMUA, newMUAnpyName, folder_pathMUA, MUA_npy_path, 
                     chunkyPathRoot, chunkyPathName, chunksList,
                     path_root, coord_root, whereUtahPath, matName, 
                     channelAreaMappingPath, rfilePATH, allsnrPATH, pathCrosstalk, 
                     snrPath, outlierRFRejectionQuantile, colsL, openEyes = False):
    
    """
    This is truly a monster function created for the sole purpose of loading monkey_L and monkey_A data
    while not using space in the main script
    """
        
    ######      COLORS & CHANNEL NUMBERS     ######
    ######      COLORS & CHANNEL NUMBERS     ######
    ######      COLORS & CHANNEL NUMBERS     ######
    
    allColors =  None
    for i in range(len(colsL)):
    
        c = np.tile(colsL[i], (64,1))#,16)
        if i == 0:
            allColors = c
        else:
            allColors = np.vstack((allColors,c))

    # Loading channel to area map
    channelAreaMap = scipy.io.loadmat(channelAreaMappingPath)
    arrayNums = channelAreaMap['arrayNums']
    areas = channelAreaMap['areas']
    channelNums = channelAreaMap['channelNums']
    
    ######      SNR & CROSSTALK      ######
    ######      SNR & CROSSTALK      ######
    ######      SNR & CROSSTALK      ######

    # loading SNR and CROSSTALK indexes for
    # if openEyes == False:

    if openEyes:
        
        snr = load_SNR_instances_openEyes(snrPath)
        snr = np.asarray(snr)
        snrArrs, snr = order_SNR_instances(snr, arrayNums, channelNums, V1ONLY=False)
        snr = snr.flatten()
        toDelete = np.where(snr < 2)
        print('                                 ')
        print(' ----- OPEN EYES SNR number ', len(toDelete[0]))
        print('snr PATH ', snrPath)
        print('snr ', snr)
        print('                                 ')

    else:
        
        # NOTA: when loading snr, the snr channels should coincide with the allSnr data in load_rfs,
        # ************ CHECK THIS *** and then only use snr as toDelete electrodes and not within load_rfs
        crossTalkers = load_crosstalk_electrodes(pathCrosstalk)
         
        snr = load_snr(snrPath)

        num_crosstalkers = crossTalkers.shape[0]
        num_noisies = snr.shape[0]
        toDelete = np.union1d(snr, crossTalkers)
        # toDelete = crossTalkers
        print('                                 ')
        print(f' ----- CLOSE EYES num crosstalkers {num_crosstalkers}, num SNR {num_noisies}, total {num_crosstalkers + num_noisies}')
        print('                                 ')

    ######      EXCLUDING V4      ######
    ######      EXCLUDING V4      ######
    ######      EXCLUDING V4      ######
    
    if EXCLUDE_V4:
        
        if monkey_LorA == 'monkey_L':
            print('Excluding V4 monkey monkey_L')
            toDelete = np.union1d(toDelete, np.arange(64,192))
            
            print('                                 ')
            print(f' ----- toDelete  V4 ', len(np.arange(64,192)) )  
            print('                                 ')
                  
        elif monkey_LorA == 'monkey_A':
            print('Excluding V4 monkey monkey_A')
            
            # #V4 arrays
            toDelete = np.union1d(toDelete, np.arange(64,128))
            toDelete = np.union1d(toDelete, np.arange(256,256+64))
                
            print('                                 ')
            print(f' ----- toDelete  V4 ', len(np.arange(64,128) + len( np.arange(256,256+64)) )  )  
            print('                                 ')

            if not openEyes:
            
                ########### %%%%%%%%%%% DEBUGGING
                toDelete = np.union1d(toDelete, np.arange(0,448+64))
                
                print('                                 ')
                print(f' ----- toDelete  V4 ', len(np.arange(0,448+64)))
                print('                                 ')

        else:
            print('WARNING: Neither monkey_L or monkey_A, check monkey_LorA variable')
            # toDelete = toDelete

    ######      RFs      ######
    ######      RFs      ######
    ######      RFs      ######

    print('                                 ')
    print(' -----    loading RFs')
    print('                                 ')

    print(rfilePATH)
    rf_list = load_rf_instances(rfilePATH)
    # ordering RFs according to their instances
    rfArrs, orderedRfs = order_RF_instances(rf_list, arrayNums, channelNums, V1ONLY=False)
    rfs = np.vstack(rfArrs)
    # rfs = np.delete(rfs, toDelete, axis = 0)
    # print('Now, while loading RFs, Im not calculating RF outliers, so nonvalidE = toDelete')
    # print('TODO: calculate RF outliers and merge and merge them with toDelete array')
    # nonvalidE = toDelete
    
    ###### RF OUTLIERS ######
    ###### RF OUTLIERS ######
    ###### RF OUTLIERS ######
    
    ### Clustering to find RF outliers
    # reshaping array to work more comfy
    rfs = rfs.reshape(rfs.shape[0],2)
    
    # Now, we will find RF ouliers
    allow_single_cluster = True
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, allow_single_cluster=allow_single_cluster).fit(rfs)
    clusterer.outlier_scores_

    threshold = pd.Series(clusterer.outlier_scores_).quantile(outlierRFRejectionQuantile)
    outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
    # print(' RF outlierRejectionQuantile : ', outlierRFRejectionQuantile)
    # print('Outliers = ', outliers)
    # es_toDelete = np.union1d(outliers, idxSNR)
    es_toDelete = outliers
    print('                                 ')
    print(' -----   RF outliers: ', len(outliers))
    print('                                 ')
    if toDelete.any() != None:
        # print('Deleting also some custom channels ', str(toDelete))
        es_toDelete = np.union1d(es_toDelete, toDelete)
    # deleting rfs
    rfs = np.delete(rfs, es_toDelete, axis = 0)
    # plt.figure(dpi=180)
    # plt.scatter(rfs[:,0], rfs[:,1])
    ### END Clustering to find RF outliers
    nonvalidE = es_toDelete

    print('                                 ')
    print(' ----- Total nonvalidE, ', len(nonvalidE) )
    print('                                 ')
    ######      UTAH      ######
    ######      UTAH      ######
    ######      UTAH      ######
    utahGood = load_utah_XING(whereUtahPath,matName,allColors,PLOT=False)
    utah = np.delete(utahGood,nonvalidE, axis = 0)
    # translating rfs to first quadrant and normalizing rf position values
    print(' not normalizing nor translating RFs ')
    #rfs[:,1] = rfs[:,1] + np.abs(rfs[:,1].min())
    #rfs /= rfs.max()
    rfs_distances = euclidean_distances(rfs)
    rfs_distances /= rfs_distances.max()

    # translating rfs to first quadrant and normalizing rf position values
    utah[:,1] = utah[:,1] + np.abs(utah[:,1].min())
    utahMAX = utah.max()
    utah /= utahMAX
    utah_distances = euclidean_distances(utah)
    utahEUmax = utah_distances.max()
    utah_distances /= utahEUmax

    ######      LFP      ######
    ######      LFP      ######
    ######      LFP      ######

    # now, just load the chunk of data you want.
    # Loading all LFP instances

    print('                                 ')
    print(' -----   USING CHUNK ' + str(CHUNKS) + '10 MATLAB INDEX')
    print('                                 ')
                  
    lfp = load_lfp_chunks(chunkyPathRoot = chunkyPathRoot,
                        chunkyPathName = chunkyPathName,
                        chunksList = CHUNKS, normalize = STANDARIZE_LFP)
    
    #% Creating array index, separating arrays, getting some useful index...
    lfp, raw_arrays, anchorIndices, colsL, arrayIndexV1, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, orig_IDx = create_array_index(lfp,colsL,num_anchors_perArray, nonvalidE)
    
    #anchor lfp channels that will be used for regression analysis
    anchorLFP = lfp[anchorIndices]
    anchorUtah = utah[anchorIndices]
    anchorUtahDistances = euclidean_distances(anchorUtah)
    anchorUtahDistances /= utahEUmax # its important to divide them by the same number as the big Utah matrix, otherwise the regression
    # analysis will not use numbers which are coherent with the rest of the data
    
    ######      MUA      ######
    ######      MUA      ######
    ######      MUA      ######
    
    #% Loading MUA
    #### load and order MUA (do only once)
    if newMUA:
        
        folder_path = folder_pathMUA       #r'LFP/eye_closed_data_MUA/'

        if openEyes:
            #load MUA and order it, then create a numpy array 
            muaList = load_MUA_instances_openEyes(folder_path, chunks = [c-1 for c in CHUNKS])
            
            MUA, hola = order_LFP_instances(muaList,arrayNums,channelNums,V1ONLY=False)
            MUAdict =  {
              "MUA": MUA}
            np.save(newMUAnpyName, MUAdict)
            
        else:
            #load MUA and order it, then create a numpy array 
            muaList = load_MUA_instances(folder_path, chunks = [c-1 for c in CHUNKS])
            MUA, hola = order_LFP_instances(muaList,arrayNums,channelNums,V1ONLY=False)
            MUAdict =  {
              "MUA": MUA}
            np.save(newMUAnpyName, MUAdict)
    
    else:
        
        #load MUA from a previous numpy array
        try:
            MUA = np.load(MUA_npy_path, allow_pickle = True).item()["MUA"][0]
        except:
            MUA = np.load(MUA_npy_path, allow_pickle = True)#.item()
            #MUA = np.load(MUA_npy_path, allow_pickle = True).item()["MUA"]
        
        
    ######      RETURN      ######
    ######      RETURN      ######
    ######      RETURN      ######
    
    return utahGood, utahMAX, nonvalidE,rfs,rfs_distances,utah,utah_distances,lfp, raw_arrays, anchorIndices, colsL, arrayIndexV1, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, orig_IDx,anchorLFP,anchorUtah,anchorUtahDistances,MUA
 
    

def load_monkey_data_SAFE_COPY(monkey_LorA,EXCLUDE_V4, CHUNKS, STANDARIZE_LFP, num_anchors_perArray,
                     newMUA, newMUAnpyName, folder_pathMUA, MUA_npy_path, 
                     chunkyPathRoot, chunkyPathName, chunksList,
                     path_root, coord_root, whereUtahPath, matName, 
                     channelAreaMappingPath, rfilePATH, allsnrPATH, pathCrosstalk, 
                     snrPath, outlierRFRejectionQuantile, colsL, openEyes = False):
    
    """
    This is truly a monster function created for the sole purpose of loading monkey_L and monkey_A data
    while not using space in the main script
    """
    
    ######      COLORS & CHANNEL NUMBERS     ######
    ######      COLORS & CHANNEL NUMBERS     ######
    ######      COLORS & CHANNEL NUMBERS     ######
    
    
    allColors =  None
    for i in range(len(colsL)):
    
        c = np.tile(colsL[i], (64,1))#,16)
        if i == 0:
            allColors = c
        else:
            allColors = np.vstack((allColors,c))

    # Loading channel to area map
    channelAreaMap = scipy.io.loadmat(channelAreaMappingPath)
    arrayNums = channelAreaMap['arrayNums']
    areas = channelAreaMap['areas']
    channelNums = channelAreaMap['channelNums']
    
    
    ######      SNR & CROSSTALK      ######
    ######      SNR & CROSSTALK      ######
    ######      SNR & CROSSTALK      ######

    
    # loading SNR and CROSSTALK indexes for
    if openEyes == False:
        crossTalkers = load_crosstalk_electrodes(pathCrosstalk)
    
    # NOTA: when loading snr, the snr channels should coincide with the allSnr data in load_rfs,
    # ************ CHECK THIS *** and then only use snr as toDelete electrodes and not within load_rfs
    
    if openEyes:
        
        snr = load_SNR_instances_openEyes(snrPath)
        snr = np.asarray(snr)
        snrArrs, snr = order_SNR_instances(snr, arrayNums, channelNums, V1ONLY=False)
        snr = snr.flatten()
        toDelete = np.where(snr < 2)
        print('                                 ')
        print(' ----- OPEN EYES SNR number ', len(toDelete))
        print('                                 ')

    else:
        snr = load_snr(snrPath)

        num_crosstalkers = crossTalkers.shape[0]
        num_noisies = snr.shape[0]
        toDelete = np.union1d(snr, crossTalkers)
        # toDelete = crossTalkers
        print('                                 ')
        print(f' ----- CLOSE EYES num crosstalkers {num_crosstalkers}, num SNR {num_noisies}, total {num_crosstalkers + num_noisies}')
        print('                                 ')

        print('                                 ')
        print(f' ----- toDelete ', len(toDelete))
        print('                                 ')
        print('intersection ', np.intersect1d(snr, crossTalkers))

    if EXCLUDE_V4:
        
        if monkey_LorA == 'monkey_L':
            print('Excluding V4 monkey monkey_L')
            toDelete = np.union1d(toDelete, np.arange(64,192))
            
        elif monkey_LorA == 'monkey_L':
            print('Excluding V4 monkey monkey_A')
            
            # #V4 arrays
            toDelete = np.union1d(toDelete, np.arange(64,128))
            toDelete = np.union1d(toDelete, np.arange(256,256+64))

            if not openEyes:
            
                ########### %%%%%%%%%%% DEBUGGING
                toDelete = np.union1d(toDelete, np.arange(0,448+64))

        else:
            print('WARNING: Neither monkey_L or monkey_A, check monkey_LorA variable')
            # toDelete = toDelete




    ######      RFs      ######
    ######      RFs      ######
    ######      RFs      ######
    
    
    print('                                 ')
    print(' -----    loading RFs')
    print('                                 ')

    print(rfilePATH)
    rf_list = load_rf_instances(rfilePATH)
    
    # ordering RFs according to their instances
    rfArrs, orderedRfs = order_RF_instances(rf_list, arrayNums, channelNums, V1ONLY=False)
    rfs = np.vstack(rfArrs)
    # rfs = np.delete(rfs, toDelete, axis = 0)

    # print('Now, while loading RFs, Im not calculating RF outliers, so nonvalidE = toDelete')
    # print('TODO: calculate RF outliers and merge and merge them with toDelete array')
    # nonvalidE = toDelete
    



    ###### RF OUTLIERS ######
    ###### RF OUTLIERS ######
    ###### RF OUTLIERS ######
  
    ### Clustering to find RF outliers
    # reshaping array to work more comfy
    rfs = rfs.reshape(rfs.shape[0],2)
    
    # Now, we will find RF ouliers
    allow_single_cluster = True
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, allow_single_cluster=allow_single_cluster).fit(rfs)
    
    clusterer.outlier_scores_
    # sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
    
    threshold = pd.Series(clusterer.outlier_scores_).quantile(outlierRFRejectionQuantile)
    outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
    # print(' RF outlierRejectionQuantile : ', outlierRFRejectionQuantile)
    # print('Outliers = ', outliers)
    # es_toDelete = np.union1d(outliers, idxSNR)
    es_toDelete = outliers
    
    print('                                 ')
    print(' -----   RF outliers: ', len(outliers))
    print('                                 ')

    if toDelete.any() != None:
        # print('Deleting also some custom channels ', str(toDelete))
        es_toDelete = np.union1d(es_toDelete, toDelete)


    # deleting rfs
    rfs = np.delete(rfs, es_toDelete, axis = 0)
    # plt.figure(dpi=180)
    # plt.scatter(rfs[:,0], rfs[:,1])
    ### END Clustering to find RF outliers
    nonvalidE = es_toDelete
 
    
 
 
    print('                                 ')
    print(' -----   nonvalidE, ', len(nonvalidE) )
    print('                                 ')
    ######      UTAH      ######
    ######      UTAH      ######
    ######      UTAH      ######
    
    
    utahGood = load_utah_XING(whereUtahPath,matName,allColors,PLOT=False)
    utah = np.delete(utahGood,nonvalidE, axis = 0)
    
    # translating rfs to first quadrant and normalizing rf position values
    print(' not normalizing nor translating RFs ')
    #rfs[:,1] = rfs[:,1] + np.abs(rfs[:,1].min())
    #rfs /= rfs.max()
    
    rfs_distances = euclidean_distances(rfs)
    rfs_distances /= rfs_distances.max()

    # translating rfs to first quadrant and normalizing rf position values
    utah[:,1] = utah[:,1] + np.abs(utah[:,1].min())
    utahMAX = utah.max()
    utah /= utahMAX
    
    utah_distances = euclidean_distances(utah)
    utahEUmax = utah_distances.max()
    utah_distances /= utahEUmax
    
    # Plotting RFs and UTAH distances
    plt.figure(dpi = 180)
    plt.imshow(rfs_distances, cmap = 'afmhot')
    plt.title('RFs electrode interdistances')
    plt.colorbar()
    
    plt.figure(dpi = 180)
    plt.imshow(utah_distances, cmap = 'afmhot')
    plt.title('Utah electrode interdistances')
    plt.colorbar()
    
    
    ######      LFP      ######
    ######      LFP      ######
    ######      LFP      ######

    # now, just load the chunk of data you want.
    # Loading all LFP instances

    print('                                 ')
    print(' -----   USING CHUNK ' + str(CHUNKS) + '10 MATLAB INDEX')
    print('                                 ')
                  
    lfp = load_lfp_chunks(chunkyPathRoot = chunkyPathRoot,
                        chunkyPathName = chunkyPathName,
                        chunksList = CHUNKS, normalize = STANDARIZE_LFP)
    
    #% Creating array index, separating arrays, getting some useful index...
    lfp, raw_arrays, anchorIndices, colsL, arrayIndexV1, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, orig_IDx = create_array_index(lfp,colsL,num_anchors_perArray, nonvalidE)
    
    #anchor lfp channels that will be used for regression analysis
    anchorLFP = lfp[anchorIndices]
    anchorUtah = utah[anchorIndices]
    anchorUtahDistances = euclidean_distances(anchorUtah)
    anchorUtahDistances /= utahEUmax # its important to divide them by the same number as the big Utah matrix, otherwise the regression
    # analysis will not use numbers which are coherent with the rest of the data
    
    
    ######      MUA      ######
    ######      MUA      ######
    ######      MUA      ######

    
    #% Loading MUA
    #### load and order MUA (do only once)
    if newMUA:
        
        folder_path = folder_pathMUA       #r'LFP/eye_closed_data_MUA/'

        if openEyes:
            #load MUA and order it, then create a numpy array 
            muaList = load_MUA_instances_openEyes(folder_path, chunks = [c-1 for c in CHUNKS])
            
            MUA, hola = order_LFP_instances(muaList,arrayNums,channelNums,V1ONLY=False)
            MUAdict =  {
              "MUA": MUA}
            np.save(newMUAnpyName, MUAdict)
            
        else:
            #load MUA and order it, then create a numpy array 
            muaList = load_MUA_instances(folder_path, chunks = [c-1 for c in CHUNKS])
            MUA, hola = order_LFP_instances(muaList,arrayNums,channelNums,V1ONLY=False)
            MUAdict =  {
              "MUA": MUA}
            np.save(newMUAnpyName, MUAdict)
    
    else:
        
        #load MUA from a previous numpy array
        try:
            MUA = np.load(MUA_npy_path, allow_pickle = True).item()["MUA"][0]
        except:
            MUA = np.load(MUA_npy_path, allow_pickle = True)#.item()
            #MUA = np.load(MUA_npy_path, allow_pickle = True).item()["MUA"]
        
        
    ######      RETURN      ######
    ######      RETURN      ######
    ######      RETURN      ######
    
    return utahGood, utahMAX, nonvalidE,rfs,rfs_distances,utah,utah_distances,lfp, raw_arrays, anchorIndices, colsL, arrayIndexV1, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, orig_IDx,anchorLFP,anchorUtah,anchorUtahDistances,MUA
 

def calculate_interArrErrorCorr(utahPr,HPr,selectedChannelsIndices, colors, PLOT = False):

    """
    takes a global recovered map and calculates error for each single array of utahPr
    which IDs are contained in the selectedChannelsIndices arrays
    """
    singleCorr  = []
    singleError = []
    
    arrayIDX = np.array(selectedChannelsIndices,dtype='object')
    
    for i in range(arrayIDX.shape[0]):
        
        singleArrIDX = arrayIDX[i]
    
        # centering arrays to 0 withouth rescaling and calculating error
        utahSingle = utahPr[singleArrIDX]
        recoveredSingle = HPr[singleArrIDX]
        utahSingle -= utahSingle.mean(0)
        recoveredSingle-= recoveredSingle.mean(0)
        
        corr = calculate_corr_of_distances(utahSingle, recoveredSingle)
        error   = np.average((utahSingle - recoveredSingle) ** 2)
        
        if PLOT:
            
            alphas = np.expand_dims(np.linspace(0.05,1,singleArrIDX.shape[0]),-1)
            arrayColorsWAlpha = np.hstack((arrayIndexV1[singleArrIDX],alphas))
            
            fig =  plt.figure( dpi = 180)
            plt.scatter(utahSingle[:, 0], utahSingle[:, 1],s=50, cmap='Spectral', 
                          alpha = 1, 
                          edgecolor="black",
                          marker = 'x',
                          c = 'black') # c=arrayIndexV1,
            plt.scatter(recoveredSingle[:, 0], recoveredSingle[:, 1],s=80, cmap='Spectral', 
                          # alpha = 1, 
                          edgecolor="black",
                          c = arrayColorsWAlpha) # c=arrayIndexV1,
        else:
            None
            
        singleCorr.append(corr)
        singleError.append(error)
        
    singleCorr = np.array(singleCorr)
    singleError = np.array(singleError)
    
    return singleCorr,singleError


def calculate_interArrErrorCorrNew(utahPr,HPr,selectedChannelsIndices, colors, PLOT = False):

    """
    takes a global recovered map and calculates error for each single array of utahPr
    which IDs are contained in the selectedChannelsIndices arrays
    """
    singleCorr  = []
    singleError = []
    
    arrayIDX = np.array(selectedChannelsIndices, dtype=object)
    
    for i in range(arrayIDX.shape[0]):
        
        singleArrIDX = arrayIDX[i]
    
        # centering arrays to 0 withouth rescaling and calculating error
        utahSingle = utahPr[singleArrIDX]
        recoveredSingle = HPr[singleArrIDX]
        utahSingle -= utahSingle.mean(0)
        recoveredSingle-= recoveredSingle.mean(0)
        
        corr = calculate_corr_of_distances(utahSingle, recoveredSingle)
        error   = np.average((utahSingle - recoveredSingle) ** 2)
        
        if PLOT:
            
            alphas = np.expand_dims(np.linspace(0.05,1,singleArrIDX.shape[0]),-1)
            arrayColorsWAlpha = np.hstack((colors[singleArrIDX],alphas))
            
            fig =  plt.figure( dpi = 180)
            plt.scatter(utahSingle[:, 0], utahSingle[:, 1],s=50, cmap='Spectral', 
                          alpha = 1, 
                          edgecolor="black",
                          marker = 'x',
                          c = 'black') # c=arrayIndexV1,
            plt.scatter(recoveredSingle[:, 0], recoveredSingle[:, 1],s=80, cmap='Spectral', 
                          # alpha = 1, 
                          edgecolor="black",
                          c = arrayColorsWAlpha) # c=arrayIndexV1,
        else:
            None
            
        singleCorr.append(corr)
        singleError.append(error)
        
    singleCorr = np.array(singleCorr)
    singleError = np.array(singleError)
    
    return singleCorr,singleError


def plot_interArrayCorrelation(singleArrCorrs_MDS,
                               singleArrCorrs_PCA,
                               singleArrCorrs_UMAP,
                               arrayNumberV1,
                               colorSingleArray):
    
    alphas = np.linspace(0.4,1,singleArrCorrs_MDS.shape[0])
    
    uniArrays = np.unique(arrayNumberV1).astype('uint32') - 1
    
    x_ax = range(singleArrCorrs_MDS.shape[1])
    plt.figure(dpi=180)
    plt.title('Inter-array inter distances correlation MDS')
    for i in range(singleArrCorrs_MDS.shape[0]):
        plt.scatter(x_ax,singleArrCorrs_MDS[i,:], 
                    color = colorSingleArray[uniArrays], s = 90,
                    edgecolor="black",
                    alpha = alphas[i])
    
    y = singleArrCorrs_MDS.mean(0)
    yerr = singleArrCorrs_MDS.std(0)
    plt.errorbar(x_ax, y, yerr=yerr, label = 'mean and std',fmt='',
                 linewidth=1, color = 'black', marker = '_', ls='none') 
    plt.legend()
    plt.xlabel('Array number')
    plt.ylabel('Correlation MDS')
    plt.ylim(0,1)
    
    #
    plt.figure(dpi=180)
    plt.title('Inter-array inter distances correlation PCA')
    for i in range(singleArrCorrs_PCA.shape[0]):
        plt.scatter(x_ax,singleArrCorrs_PCA[i,:], 
                    color = colorSingleArray[uniArrays], s = 90,
                    edgecolor="black",
                    alpha = alphas[i])
        
    y = singleArrCorrs_PCA.mean(0)
    yerr = singleArrCorrs_PCA.std(0)
    plt.errorbar(x_ax, y, yerr=yerr, label = 'mean and std',fmt='',
                 linewidth=1, color = 'black', marker = '_', ls='none') 
    plt.legend()
    plt.xlabel('Array number')
    plt.ylabel('Correlation PCA')
    plt.ylim(0,1)
    
    #
    plt.figure(dpi=180)
    plt.title('Inter-array inter distances correlation UMAP')
    for i in range(singleArrCorrs_UMAP.shape[0]):
        plt.scatter(x_ax,singleArrCorrs_UMAP[i,:], 
                    color = colorSingleArray[uniArrays], s = 90,
                    edgecolor="black",
                    alpha = alphas[i])
    
    y = singleArrCorrs_UMAP.mean(0)
    yerr = singleArrCorrs_UMAP.std(0)
    plt.errorbar(x_ax, y, yerr=yerr, label = 'mean and std',fmt='',
                 linewidth=1, color = 'black', marker = '_', ls='none') 
    plt.legend()
    plt.xlabel('Array number')
    plt.ylabel('Correlation UMAP')
    plt.ylim(0,1)

def plot_interArrayError(singleArrErrors_MDS,
                               singleArrErrors_PCA,
                               singleArrErrors_UMAP,
                               arrayNumberV1,
                               colorSingleArray):
    
    alphas = np.linspace(0.4,1,singleArrErrors_MDS.shape[0])
    
    uniArrays = np.unique(arrayNumberV1).astype('uint32') - 1
    
    x_ax = range(singleArrErrors_MDS.shape[1])
    plt.figure(dpi=180)
    plt.title('Inter-array inter distances error MDS')
    for i in range(singleArrErrors_MDS.shape[0]):
        plt.scatter(x_ax,singleArrErrors_MDS[i,:], 
                    color = colorSingleArray[uniArrays], s = 90,
                    edgecolor="black",
                    alpha = alphas[i])
    
    y = singleArrErrors_MDS.mean(0)
    yerr = singleArrErrors_MDS.std(0)
    plt.errorbar(x_ax, y, yerr=yerr, label = 'mean and std',fmt='',
                 linewidth=1, color = 'black', marker = '_', ls='none') 
    plt.legend()
    plt.xlabel('Array number')
    plt.ylabel('Error MDS')
    plt.ylim(0,0.00025)
    
    #
    plt.figure(dpi=180)
    plt.title('Inter-array inter distances error PCA')
    for i in range(singleArrErrors_PCA.shape[0]):
        plt.scatter(x_ax,singleArrErrors_PCA[i,:], 
                    color = colorSingleArray[uniArrays], s = 90,
                    edgecolor="black",
                    alpha = alphas[i])
        
    y = singleArrErrors_PCA.mean(0)
    yerr = singleArrErrors_PCA.std(0)
    plt.errorbar(x_ax, y, yerr=yerr, label = 'mean and std',fmt='',
                 linewidth=1, color = 'black', marker = '_', ls='none') 
    plt.legend()
    plt.xlabel('Array number')
    plt.ylabel('Error PCA')
    plt.ylim(0,0.00025)
    
    #
    plt.figure(dpi=180)
    plt.title('Inter-array inter distances error UMAP')
    for i in range(singleArrErrors_UMAP.shape[0]):
        plt.scatter(x_ax,singleArrErrors_UMAP[i,:], 
                    color = colorSingleArray[uniArrays], s = 90,
                    edgecolor="black",
                    alpha = alphas[i])
    
    y = singleArrErrors_UMAP.mean(0)
    yerr = singleArrErrors_UMAP.std(0)
    plt.errorbar(x_ax, y, yerr=yerr, label = 'mean and std',fmt='',
                 linewidth=1, color = 'black', marker = '_', ls='none') 
    plt.legend()
    plt.xlabel('Array number')
    plt.ylabel('Error UMAP')
    plt.ylim(0,0.00025)
    


def loadMonkey3sessionsData(monkey,STANDARIZE_LFP,EXCLUDE_V4, basePath, whereUtahPath, channelAreaMappingPath, deleteElecsPath):
    
    
    """
    This doeas as in paper_figureV8_loading3sessions2monkeysdata.py but in a function
    
    basically it picks the utah, RFs, LFP and MUA data, colors etc. for that monkey
    
    """
    
    colsL = [(1.0000, 0,  0,), (1.0000, 0.3750, 0), (1.0000, 0.7500, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706, 0.0745),
             (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0, 0.2500, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750)]
    colsA = [(1.0000, 0, 0), (1.0000, 0.3750, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (1.0000, 0.7500, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706,0.0745), 
             (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750), (0, 0, 0)]
    
    # Creating colors for every electrode
    allColors =  None
    for i in range(len(colsL)):
    
        c = np.tile(colsL[i], (64,1))#,16)
        if i == 0:
            allColors = c
        else:
            allColors = np.vstack((allColors,c))
    
    # Loading channel to area map
    channelAreaMap = scipy.io.loadmat(channelAreaMappingPath)
    arrayNums = channelAreaMap['arrayNums']
    areas = channelAreaMap['areas']
    channelNums = channelAreaMap['channelNums']

    ########### *** DELETING ELECTRODES IS NOW OVERRIDED BY LOADING NONVALID ELECTRODES LIST, ***
    ########### ***  GENERATED IN paper_figuresV7_currentCopy.py *** it is also in later versions
    ########### ***  STORED IN 'LFP-RFs/data_results/deletedElectrodesDictionary/' *** 

    #%
    
    print('Loading RFs...')
    rfilePATH = basePath + monkey + '/RFS/'
    rf_list = load_rf_instances(rfilePATH)
    
    # Order RFs according to their instances
    rfArrs, orderedRfs = order_RF_instances(rf_list, arrayNums, channelNums, V1ONLY=False)
    rfs = np.vstack(rfArrs)
    
    print('Now, while loading RFs, I am not calculating RF outliers, so nonvalidE = toDelete')
    print('TODO: Calculate RF outliers and merge them with toDelete array')
    
    # Load utah arrays
    # whereUtahPath = r'LFP/coordinates_of_electrodes_on_cortex_using_photos_of_arrays/'
    matName = 'allPixelIDs_monkey_L.mat' if monkey == 'monkey_L' else 'allPixelIDs_monkey_L.mat'
    utah = load_utah_XING(whereUtahPath, matName, allColors, PLOT=False)
    
    # print('NOT NORMALIZING RFS USING MAX BECAUSE MAX IS AN EXTREME VALUE')
    
    # Compute euclidean distances for RFs
    rfs_distances = euclidean_distances(rfs)
    
    # Translate and normalize utah electrode positions
    utah[:,1] += np.abs(utah[:,1].min())
    UTAHMAX = utah.max()
    utah /= UTAHMAX
    
    # Compute euclidean distances for utah
    utah_distances = euclidean_distances(utah)
    utah_distances /= utah_distances.max()
    
    
    # Compute 2 standard deviations for rfs_distances and utah_distances
    rfs_q1 = np.percentile(rfs_distances, 5)
    rfs_q3 = np.percentile(rfs_distances, 95)
    utah_std2 = 5 * np.std(utah_distances)
    
    # Plotting RFs
    plt.figure(dpi=300)
    plt.scatter(rfs[:,0], rfs[:,1], c=allColors)
    plt.ylim(-200, 100)
    plt.xlim(-100, 200)
    
    # Plotting RF and utah inter-distances
    fig, axs = plt.subplots(2, dpi=180)
    
    # Use 2 standard deviations for the colorbar range in imshow
    img1 = axs[0].imshow(rfs_distances, cmap='afmhot', vmin=rfs_q1, vmax=rfs_q3)
    axs[0].set_title('RFs electrode interdistances')
    fig.colorbar(img1, ax=axs[0])
    
    img2 = axs[1].imshow(utah_distances, cmap='afmhot', vmin=0, vmax=utah_std2)
    axs[1].set_title('Utah electrode interdistances')
    fig.colorbar(img2, ax=axs[1])
    
    plt.tight_layout()
    
    arrayIndexV1 = allColors
    
    #% Loading Bad Channels
    ########### DELETING ELECTRODES IS NOW OVERRIDED BY LOADING NONVALID ELECTRODES LIST,
    ########### GENERATED IN paper_figuresV7_currentCopy.py
    ########### STORED IN 'LFP-RFs/data_results/deletedElectrodesDictionary/'
    if monkey == 'monkey_L':
        TOLOAD = 'monkey_LClosedEyes'
    else:
        TOLOAD = 'monkey_LClosedEyes'
        
    with open(os.path.join(deleteElecsPath, 'subjectCondition_NonvalidElectrodesList.pkl'), 'rb') as f:
        badElectrodesDict = pickle.load(f)
    subjectConditionList = np.array( badElectrodesDict['subject'] )
    nonvalidEList = badElectrodesDict['nonvalidElectrodesList']
    idxNonvalid = np.where( subjectConditionList == TOLOAD)[0][0]
    nonvalidE = nonvalidEList[idxNonvalid]
    
    #% LOADING ALL LFP AND MUA DATA (ignoring the first 4 seconds of each chunk of data)
    
    # EYES CLOSED
    basePath + monkey + '/LFP/'
    LFP = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_' + monkey + '.npy', allow_pickle=True)
    print(LFP.shape)
    basePath + monkey + '/MUA/'
    # LFP_gamma = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_gamma_' + monkey + '.npy', allow_pickle=True)

    MUABINNED = np.load(basePath + monkey + '/MUA/allOrderedMUA_binsize_2_ignore4secs_' + monkey + '.npy', allow_pickle = True)
    print(MUABINNED.shape)

    #% Creating array index, separating arrays, getting some useful index...
    num_anchors_perArray = 2

    # lfp, raw_arrays, anchorIndices, colsL, allColors, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, orig_IDx = create_array_index_new(LFP[:,0:100],colsL,num_anchors_perArray, nonvalidE)
    # using random array to obtain 1024 colors, deteling the bad ones later
    lfp, raw_arrays, anchorIndices, colsL, allColors, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, orig_IDx = create_array_index_new(np.random.rand(1024, 10),colsL,num_anchors_perArray, nonvalidE)
    
    del lfp, raw_arrays
    
    #anchor lfp channels that will be used for regression analysis
    anchorLFP = LFP[anchorIndices]
    anchorUtah = utah[anchorIndices]
    anchorUtahDistances = euclidean_distances(anchorUtah)
    # anchorUtahDistances /= utahEUmax # its important to divide them by the same number as the big Utah matrix, otherwise the regression
    
    utah = np.delete((utah),nonvalidE,axis=0)
    rfs = np.delete((rfs),nonvalidE,axis=0)
    
    # LFP AND MUABINNED WITH OUTLIER CHUNKS REMOVED ALREADY HAVE THE NONVALID ELECTRODES REMOVED 
    
    LFP = np.delete((LFP),nonvalidE,axis=0)
    # LFP_gamma = np.delete((LFP_gamma),nonvalidE,axis=0)
    MUABINNED = np.delete((MUABINNED),nonvalidE,axis=0)
    
    colors = allColors
    arrayIDX = np.array(selectedChannelsIndices, dtype=object)
    
    #createAlphas for plotting
    colorsWAlpha = []
    for j in range(arrayIDX.shape[0]):
        
        singleArrIDX = arrayIDX[j]
        alphas = np.expand_dims(np.linspace(0.3,1,singleArrIDX.shape[0]),-1)
        alphas = np.power(alphas,2)
        arrayColorsWAlpha = np.hstack((allColors[singleArrIDX],alphas))
        # arrayColorsWAlpha = np.hstack((arrayIndexV1[singleArrIDX],alphas))
        colorsWAlpha.append(arrayColorsWAlpha)   
    
    colorsWAlpha = np.vstack(colorsWAlpha)


    return  LFP, MUABINNED, anchorLFP, anchorUtah, anchorUtahDistances, colors, arrayNumberV1, arrayIDX, colorsWAlpha, \
        channelAreaMap, arrayNums, areas, channelNums, rfs, rfs_distances, utah, utah_distances, \
        badElectrodesDict, subjectConditionList, nonvalidEList, idxNonvalid, \
        nonvalidE, UTAHMAX,selectedChannelsIndices
        
        

def loadMonkey3sessionsData_outliersOut(monkey,STANDARIZE_LFP,EXCLUDE_V4, basePath, whereUtahPath, channelAreaMappingPath, deleteElecsPath):
    
    """
    This doeas as in paper_figureV8_loading3sessions2monkeysdata.py but in a function
    
    basically it picks the utah, RFs, LFP and MUA data, colors etc. for that monkey
    
    """
     
    colsL = [(1.0000, 0,  0,), (1.0000, 0.3750, 0), (1.0000, 0.7500, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706, 0.0745),
             (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0, 0.2500, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750)]
    colsA = [(1.0000, 0, 0), (1.0000, 0.3750, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (1.0000, 0.7500, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706,0.0745), 
             (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750), (0, 0, 0)]
    
    # Creating colors for every electrode
    allColors =  None
    for i in range(len(colsL)):
    
        c = np.tile(colsL[i], (64,1))#,16)
        if i == 0:
            allColors = c
        else:
            allColors = np.vstack((allColors,c))
    
    # Loading channel to area map
    channelAreaMap = scipy.io.loadmat(channelAreaMappingPath)
    arrayNums = channelAreaMap['arrayNums']
    areas = channelAreaMap['areas']
    channelNums = channelAreaMap['channelNums']

    print('Loading RFs...')
    rfilePATH = basePath + monkey + '/RFS/'
    rf_list = load_rf_instances(rfilePATH)
    
    # Order RFs according to their instances
    rfArrs, orderedRfs = order_RF_instances(rf_list, arrayNums, channelNums, V1ONLY=False)
    rfs = np.vstack(rfArrs)
    
    print('Now, while loading RFs, I am not calculating RF outliers, so nonvalidE = toDelete')
    print('TODO: Calculate RF outliers and merge them with toDelete array')
    
    # Load utah arrays
    # whereUtahPath = r'LFP/coordinates_of_electrodes_on_cortex_using_photos_of_arrays/'
    matName = 'allPixelIDs_monkey_L.mat' if monkey == 'monkey_L' else 'allPixelIDs_monkey_L.mat'
    utah = load_utah_XING(whereUtahPath, matName, allColors, PLOT=False)
    
    # print('NOT NORMALIZING RFS USING MAX BECAUSE MAX IS AN EXTREME VALUE')
    
    # Compute euclidean distances for RFs
    rfs_distances = euclidean_distances(rfs)
    
    # Translate and normalize utah electrode positions
    utah[:,1] += np.abs(utah[:,1].min())
    UTAHMAX = utah.max()
    utah /= UTAHMAX
    
    # Compute euclidean distances for utah
    utah_distances = euclidean_distances(utah)
    utah_distances /= utah_distances.max()
    
    
    # Compute 2 standard deviations for rfs_distances and utah_distances
    rfs_q1 = np.percentile(rfs_distances, 5)
    rfs_q3 = np.percentile(rfs_distances, 95)
    utah_std2 = 5 * np.std(utah_distances)
    
    # Plotting RFs
    plt.figure(dpi=300)
    plt.scatter(rfs[:,0], rfs[:,1], c=allColors)
    plt.ylim(-200, 100)
    plt.xlim(-100, 200)
    
    # Plotting RF and utah inter-distances
    fig, axs = plt.subplots(2, dpi=180)
    
    # Use 2 standard deviations for the colorbar range in imshow
    img1 = axs[0].imshow(rfs_distances, cmap='afmhot', vmin=rfs_q1, vmax=rfs_q3)
    axs[0].set_title('RFs electrode interdistances')
    fig.colorbar(img1, ax=axs[0])
    
    img2 = axs[1].imshow(utah_distances, cmap='afmhot', vmin=0, vmax=utah_std2)
    axs[1].set_title('Utah electrode interdistances')
    fig.colorbar(img2, ax=axs[1])
    
    plt.tight_layout()
    
    arrayIndexV1 = allColors
    
    #% Loading Bad Channels
    ########### DELETING ELECTRODES IS NOW OVERRIDED BY LOADING NONVALID ELECTRODES LIST,
    ########### GENERATED IN paper_figuresV7_currentCopy.py
    ########### STORED IN 'LFP-RFs/data_results/deletedElectrodesDictionary/'
    
    if monkey == 'monkey_L':
        TOLOAD = 'monkey_LClosedEyes'
    else:
        TOLOAD = 'monkey_LClosedEyes'
        
    with open(os.path.join(deleteElecsPath, 'subjectCondition_NonvalidElectrodesList.pkl'), 'rb') as f:
        badElectrodesDict = pickle.load(f)
    subjectConditionList = np.array( badElectrodesDict['subject'] )
    nonvalidEList = badElectrodesDict['nonvalidElectrodesList']
    idxNonvalid = np.where( subjectConditionList == TOLOAD)[0][0]
    nonvalidE = nonvalidEList[idxNonvalid]
    
    #% LOADING ALL LFP AND MUA DATA (ignoring the first 4 seconds of each chunk of data)
    # EYES CLOSED
    basePath + monkey + '/LFP/'
    LFP = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_' + monkey + '_outlierChunks_removed.npy', allow_pickle=True)
    print(LFP.shape)
    basePath + monkey + '/MUA/'
    # LFP_gamma = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_gamma_' + monkey + '.npy', allow_pickle=True)
    MUABINNED = np.load(basePath + monkey + '/MUA/allOrderedMUA_binsize_2_ignore4secs_' + monkey + '_outlierChunks_removed.npy', allow_pickle = True)
    print(MUABINNED.shape)

    #% Creating array index, separating arrays, getting some useful index...
    num_anchors_perArray = 2

    # using random array to obtain 1024 colors, deteling the bad ones later
    lfp, raw_arrays, anchorIndices, colsL, allColors, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, orig_IDx = create_array_index_new(np.random.rand(1024, 10),colsL,num_anchors_perArray, nonvalidE)
    
    del lfp, raw_arrays
    
    #anchor lfp channels that will be used for regression analysis
    anchorLFP = LFP[anchorIndices]
    anchorUtah = utah[anchorIndices]
    anchorUtahDistances = euclidean_distances(anchorUtah)
    
    utah = np.delete((utah),nonvalidE,axis=0)
    rfs = np.delete((rfs),nonvalidE,axis=0)
    
    # LFP AND MUABINNED WITH OUTLIER CHUNKS REMOVED ALREADY HAVE THE NONVALID ELECTRODES REMOVED 
    # LFP = np.delete((LFP),nonvalidE,axis=0)
    # LFP_gamma = np.delete((LFP_gamma),nonvalidE,axis=0)
    # MUABINNED = np.delete((MUABINNED),nonvalidE,axis=0)
    
    colors = allColors
    arrayIDX = np.array(selectedChannelsIndices)
    
    #createAlphas for plotting
    colorsWAlpha = []
    for j in range(arrayIDX.shape[0]):
        
        singleArrIDX = arrayIDX[j]
        alphas = np.expand_dims(np.linspace(0.3,1,singleArrIDX.shape[0]),-1)
        alphas = np.power(alphas,2)
        arrayColorsWAlpha = np.hstack((allColors[singleArrIDX],alphas))
        # arrayColorsWAlpha = np.hstack((arrayIndexV1[singleArrIDX],alphas))
        colorsWAlpha.append(arrayColorsWAlpha)   
    
    colorsWAlpha = np.vstack(colorsWAlpha)

    return  LFP, MUABINNED, anchorLFP, anchorUtah, anchorUtahDistances, colors, arrayNumberV1, arrayIDX, colorsWAlpha, \
        channelAreaMap, arrayNums, areas, channelNums, rfs, rfs_distances, utah, utah_distances, \
        badElectrodesDict, subjectConditionList, nonvalidEList, idxNonvalid, \
        nonvalidE, UTAHMAX,selectedChannelsIndices
        

def loadMonkey3sessionsData_outliersOut_loadSpecificBand(monkey, STANDARIZE_LFP, EXCLUDE_V4, basePath, whereUtahPath, channelAreaMappingPath, 
                                                         deleteElecsPath, freq_band="LFP", load_MUA=False, LFP_float16 = False):

    """
    This doeas as in paper_figureV8_loading3sessions2monkeysdata.py but in a function
    
    basically it picks the utah, RFs, LFP and MUA data, colors etc. for that monkey
    
    This version of the function also loads ONLY the specified frequency band, and optionally it loads the MUABINNED data
    
    """
    basePath = os.path.normpath(basePath)
     
    colsL = [(1.0000, 0,  0,), (1.0000, 0.3750, 0), (1.0000, 0.7500, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706, 0.0745),
             (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0, 0.2500, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750)]
    colsA = [(1.0000, 0, 0), (1.0000, 0.3750, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (1.0000, 0.7500, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706,0.0745), 
             (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750), (0, 0, 0)]
    
    if monkey == 'monkey_L':
        None
    else:
        colsL = colsA
        
    # Creating colors for every electrode
    allColors =  None
    for i in range(len(colsL)):
    
        c = np.tile(colsL[i], (64,1))#,16)
 
        if i == 0:
            allColors = c
        else:
            allColors = np.vstack((allColors,c))
    
    # Loading channel to area map
    channelAreaMap = scipy.io.loadmat(channelAreaMappingPath)
    arrayNums = channelAreaMap['arrayNums']
    areas = channelAreaMap['areas']
    channelNums = channelAreaMap['channelNums']

    print('Loading RFs...')
    rfilePATH = os.path.join(basePath, monkey, 'RFS')
    
    print('rfilePATH ', rfilePATH)
    print('rfilePATH ', rfilePATH)
    print('rfilePATH ', rfilePATH)
    print('rfilePATH ', rfilePATH)
    rf_list = load_rf_instances(rfilePATH)
    
    # Order RFs according to their instances
    rfArrs, orderedRfs = order_RF_instances(rf_list, arrayNums, channelNums, V1ONLY=False)
    rfs = np.vstack(rfArrs)
    
    print('Now, while loading RFs, I am not calculating RF outliers, so nonvalidE = toDelete')
    print('TODO: Calculate RF outliers and merge them with toDelete array')
    
    # Load utah arrays
    # whereUtahPath = r'LFP/coordinates_of_electrodes_on_cortex_using_photos_of_arrays/'
    matName = 'allPixelIDs_monkey_L.mat' if monkey == 'monkey_L' else 'allPixelIDs_monkey_L.mat'
    utah = load_utah_XING(whereUtahPath, matName, allColors, PLOT=False)
    
    # print('NOT NORMALIZING RFS USING MAX BECAUSE MAX IS AN EXTREME VALUE')
    
    # Compute euclidean distances for RFs
    rfs_distances = euclidean_distances(rfs)
    
    # Translate and normalize utah electrode positions
    utah[:,1] += np.abs(utah[:,1].min())
    UTAHMAX = utah.max()
    utah /= UTAHMAX
    
    # Compute euclidean distances for utah
    utah_distances = euclidean_distances(utah)
    utah_distances /= utah_distances.max()
    
    # Compute 2 standard deviations for rfs_distances and utah_distances
    rfs_q1 = np.percentile(rfs_distances, 5)
    rfs_q3 = np.percentile(rfs_distances, 95)
    utah_std2 = 5 * np.std(utah_distances)
    
    # # Plotting RFs
    # plt.figure(dpi=300)
    # plt.scatter(rfs[:,0], rfs[:,1], c=allColors)
    # plt.ylim(-200, 100)
    # plt.xlim(-100, 200)
    
    # # Plotting RF and utah inter-distances
    # fig, axs = plt.subplots(2, dpi=180)
    
    # # Use 2 standard deviations for the colorbar range in imshow
    # img1 = axs[0].imshow(rfs_distances, cmap='afmhot', vmin=rfs_q1, vmax=rfs_q3)
    # axs[0].set_title('RFs electrode interdistances')
    # fig.colorbar(img1, ax=axs[0])
    
    # img2 = axs[1].imshow(utah_distances, cmap='afmhot', vmin=0, vmax=utah_std2)
    # axs[1].set_title('Utah electrode interdistances')
    # fig.colorbar(img2, ax=axs[1])
    
    # plt.tight_layout()
    
    arrayIndexV1 = allColors
    
    #% Loading Bad Channels
    ########### DELETING ELECTRODES IS NOW OVERRIDED BY LOADING NONVALID ELECTRODES LIST,
    ########### GENERATED IN paper_figuresV7_currentCopy.py
    ########### STORED IN 'LFP-RFs/data_results/deletedElectrodesDictionary/'
    
    if monkey == 'monkey_L':
        TOLOAD = 'monkey_LClosedEyes'
    else:
        TOLOAD = 'monkey_LClosedEyes'
        
    with open(os.path.join(deleteElecsPath, 'subjectCondition_NonvalidElectrodesList.pkl'), 'rb') as f:
        badElectrodesDict = pickle.load(f)
        
    subjectConditionList = np.array( badElectrodesDict['subject'] )
    nonvalidEList = badElectrodesDict['nonvalidElectrodesList']
    idxNonvalid = np.where( subjectConditionList == TOLOAD)[0][0]
    nonvalidE = nonvalidEList[idxNonvalid]
    
    #% LOADING ALL LFP AND MUA DATA (ignoring the first 4 seconds of each chunk of data)
    
    # EYES CLOSED
    # basePath + monkey + '/LFP/'
    # LFP = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_' + monkey + '_outlierChunks_removed.npy', allow_pickle=True)
    # print(LFP.shape)
    # basePath + monkey + '/MUA/'
    # # LFP_gamma = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_gamma_' + monkey + '.npy', allow_pickle=True)
    # MUABINNED = np.load(basePath + monkey + '/MUA/allOrderedMUA_binsize_2_ignore4secs_' + monkey + '_outlierChunks_removed.npy', allow_pickle = True)
    # print(MUABINNED.shape)


    # Conditionally load LFP data according to the specified frequency band
    # valid_freq_bands = ["LFP", "low","alpha", "beta", "gamma", "high_gamma"]
    valid_freq_bands = ["LFP", "low","alpha", "beta", "gamma", "highGamma"]
    if freq_band == 'high_gamma':
        freq_band = 'highGamma'
     
    # Mapping from given frequency bands to standardized names
    freq_band_mapping = {
    'original': 'LFP',
    'mattLow': 'low',
    'mattAlpha': 'alpha',
    'mattBeta': 'beta',
    'mattGamma': 'gamma',
    'mattHighGamma': 'highGamma'
    }

    freq_band = freq_band_mapping.get(freq_band, freq_band)     
    valid_freq_bands = ["LFP", "low", "alpha", "beta", "gamma", "highGamma"]

    if freq_band not in valid_freq_bands:
        raise ValueError(f"Invalid frequency band specified. Choose from {valid_freq_bands}")

    # LFP = np.load(basePath + monkey + f'/LFP/allOrdered{freq_band}_ignore4secs_' + monkey + '_outlierChunks_removed.npy', allow_pickle=True)
    
    if freq_band == 'LFP':
        LFP_path = os.path.join(basePath, monkey, 'LFP', f'allOrderedLFP_ignore4secs_{freq_band}_{monkey}_outlierChunks_removed_15seconds.npy')
    else:
        LFP_path = os.path.join(basePath, monkey, 'LFP', f'allOrderedLFP_ignore4secs_{freq_band}_{monkey}_outlierChunks_removed.npy')
    
    if LFP_float16 == False:
        LFP = np.load(LFP_path, allow_pickle=True)
    else:
        print('Loading LFP as float16 to reduce memory usage')
        LFP = np.load(LFP_path, allow_pickle=True).astype('float16')

    print(f"LFP.shape: {LFP.shape[0]} channels and {LFP.shape[1]} time points.")
    print(f"LFP.dtype: {LFP.dtype} ")

    
    # Conditionally load MUA data if specified
    MUABINNED = None  # Initialize to None
    if load_MUA:
        mua_path = os.path.join(
            basePath,
            monkey,
            'MUA',
            f'allOrderedMUA_binsize_2_ignore4secs_{monkey}_outlierChunks_removed.npy',
        )
        MUABINNED = np.load(mua_path, allow_pickle=True)
        print(MUABINNED.shape)

    #% Creating array index, separating arrays, getting some useful index...
    num_anchors_perArray = 2

    # using random array to obtain 1024 colors, deteling the bad ones later
    lfp, raw_arrays, anchorIndices, colsL, allColors, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, orig_IDx = create_array_index_new(np.random.rand(1024, 10),colsL,num_anchors_perArray, nonvalidE)
 
    del lfp, raw_arrays
    
    #anchor lfp channels that will be used for regression analysis
    anchorLFP = LFP[anchorIndices]
    anchorUtah = utah[anchorIndices]
    anchorUtahDistances = euclidean_distances(anchorUtah)
    
    utah = np.delete((utah),nonvalidE,axis=0)
    rfs = np.delete((rfs),nonvalidE,axis=0)
    
    # LFP AND MUABINNED WITH OUTLIER CHUNKS REMOVED ALREADY HAVE THE NONVALID ELECTRODES REMOVED 
    # LFP = np.delete((LFP),nonvalidE,axis=0)
    # LFP_gamma = np.delete((LFP_gamma),nonvalidE,axis=0)
    # MUABINNED = np.delete((MUABINNED),nonvalidE,axis=0)
    
    colors = allColors
    arrayIDX = np.array(selectedChannelsIndices)
    
    #createAlphas for plotting
    colorsWAlpha = []
    for j in range(arrayIDX.shape[0]):
        
        singleArrIDX = arrayIDX[j]
        alphas = np.expand_dims(np.linspace(0.3,1,singleArrIDX.shape[0]),-1)
        alphas = np.power(alphas,2)
        arrayColorsWAlpha = np.hstack((allColors[singleArrIDX],alphas))
        # arrayColorsWAlpha = np.hstack((arrayIndexV1[singleArrIDX],alphas))
        colorsWAlpha.append(arrayColorsWAlpha)   
    
    colorsWAlpha = np.vstack(colorsWAlpha)

    return  LFP, MUABINNED, anchorLFP, anchorUtah, anchorUtahDistances, colors, arrayNumberV1, arrayIDX, colorsWAlpha, \
        channelAreaMap, arrayNums, areas, channelNums, rfs, rfs_distances, utah, utah_distances, \
        badElectrodesDict, subjectConditionList, nonvalidEList, idxNonvalid, \
        nonvalidE, UTAHMAX,selectedChannelsIndices
        
        

    
def loadMonkey3sessionsData_15secsOnly(monkey,STANDARIZE_LFP,EXCLUDE_V4, basePath, whereUtahPath, channelAreaMappingPath, deleteElecsPath):
    
    
    """
    This doeas as in paper_figureV8_loading3sessions2monkeysdata.py but in a function
    
    basically it picks the utah, RFs, LFP and MUA data, colors etc. for that monkey
    
    
    the only difference with the other function is this one loads a 15 seconds sample from previously saved 
    LFP data thus its faster 
    
    """
    
    
    colsL = [(1.0000, 0,  0,), (1.0000, 0.3750, 0), (1.0000, 0.7500, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706, 0.0745),
             (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0, 0.2500, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750)]
    colsA = [(1.0000, 0, 0), (1.0000, 0.3750, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (1.0000, 0.7500, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706,0.0745), 
             (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750), (0, 0, 0)]
    
    # Creating colors for every electrode
    allColors =  None
    for i in range(len(colsL)):
    
        c = np.tile(colsL[i], (64,1))#,16)
        if i == 0:
            allColors = c
        else:
            allColors = np.vstack((allColors,c))
    
    # Loading channel to area map
    channelAreaMap = scipy.io.loadmat(channelAreaMappingPath)
    arrayNums = channelAreaMap['arrayNums']
    areas = channelAreaMap['areas']
    channelNums = channelAreaMap['channelNums']

    ########### *** DELETING ELECTRODES IS NOW OVERRIDED BY LOADING NONVALID ELECTRODES LIST, ***
    ########### ***  GENERATED IN paper_figuresV7_currentCopy.py *** it is also in later versions
    ########### ***  STORED IN 'NIN/LFP-RFs/data_results/deletedElectrodesDictionary/' *** 

    #%
    
    print('Loading RFs...')
    rfilePATH = basePath + monkey + '/RFS/'
    rf_list = load_rf_instances(rfilePATH)
    
    # Order RFs according to their instances
    rfArrs, orderedRfs = order_RF_instances(rf_list, arrayNums, channelNums, V1ONLY=False)
    rfs = np.vstack(rfArrs)
    
    print('Now, while loading RFs, I am not calculating RF outliers, so nonvalidE = toDelete')
    print('TODO: Calculate RF outliers and merge them with toDelete array')
    
    # Load utah arrays
    # whereUtahPath = r'LFP/coordinates_of_electrodes_on_cortex_using_photos_of_arrays/'
    matName = 'allPixelIDs_monkey_L.mat' if monkey == 'monkey_L' else 'allPixelIDs_monkey_L.mat'
    utah = load_utah_XING(whereUtahPath, matName, allColors, PLOT=False)
    
    # print('NOT NORMALIZING RFS USING MAX BECAUSE MAX IS AN EXTREME VALUE')
    
    # Compute euclidean distances for RFs
    rfs_distances = euclidean_distances(rfs)
    
    # Translate and normalize utah electrode positions
    utah[:,1] += np.abs(utah[:,1].min())
    UTAHMAX = utah.max()
    utah /= UTAHMAX
    
    # Compute euclidean distances for utah
    utah_distances = euclidean_distances(utah)
    utah_distances /= utah_distances.max()
    
    
    # Compute 2 standard deviations for rfs_distances and utah_distances
    rfs_q1 = np.percentile(rfs_distances, 5)
    rfs_q3 = np.percentile(rfs_distances, 95)
    utah_std2 = 5 * np.std(utah_distances)
    
    # Plotting RFs
    plt.figure(dpi=300)
    plt.scatter(rfs[:,0], rfs[:,1], c=allColors)
    plt.ylim(-200, 100)
    plt.xlim(-100, 200)
    
    # Plotting RF and utah inter-distances
    fig, axs = plt.subplots(2, dpi=180)
    
    # Use 2 standard deviations for the colorbar range in imshow
    img1 = axs[0].imshow(rfs_distances, cmap='afmhot', vmin=rfs_q1, vmax=rfs_q3)
    axs[0].set_title('RFs electrode interdistances')
    fig.colorbar(img1, ax=axs[0])
    
    img2 = axs[1].imshow(utah_distances, cmap='afmhot', vmin=0, vmax=utah_std2)
    axs[1].set_title('Utah electrode interdistances')
    fig.colorbar(img2, ax=axs[1])
    
    plt.tight_layout()
    
    arrayIndexV1 = allColors
    
    #% Loading Bad Channels
    ########### DELETING ELECTRODES IS NOW OVERRIDED BY LOADING NONVALID ELECTRODES LIST,
    ########### GENERATED IN paper_figuresV7_currentCopy.py
    ########### STORED IN 'LFP-RFs/data_results/deletedElectrodesDictionary/'
    if monkey == 'monkey_L':
        TOLOAD = 'monkey_LClosedEyes'
    else:
        TOLOAD = 'monkey_LClosedEyes'
        
    with open(os.path.join(deleteElecsPath, 'subjectCondition_NonvalidElectrodesList.pkl'), 'rb') as f:
        badElectrodesDict = pickle.load(f)
    subjectConditionList = np.array( badElectrodesDict['subject'] )
    nonvalidEList = badElectrodesDict['nonvalidElectrodesList']
    idxNonvalid = np.where( subjectConditionList == TOLOAD)[0][0]
    nonvalidE = nonvalidEList[idxNonvalid]
    
    #% LOADING ALL LFP AND MUA DATA (ignoring the first 4 seconds of each chunk of data)
    
    # EYES CLOSED
    basePath + monkey + '/LFP/'
    # 1 Hz high pass filtered data 
    # LFP = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_HighPass1Hz_' + monkey + '.npy', allow_pickle=True)
    LFP = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_' + monkey + '_15seconds.npy', allow_pickle=True)
    # LFP = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_' + 'HighPass1Hz_' + monkey + '.npy', allow_pickle=True)
    
    print(LFP.shape)
    basePath + monkey + '/MUA/'
    # LFP_gamma = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_gamma_' + monkey + '.npy', allow_pickle=True)
    # MUA = np.load(basePath + monkey + '/MUA/allOrderedMUA_ignore4secs' + monkey + '.npy', allow_pickle=True)
    # MUABINNED = np.load(basePath + monkey + '/MUA/allOrderedMUA_BINNED_ignore4secs_' + monkey + '.npy', allow_pickle = True)
    # MUABINNED = np.load(basePath + monkey + '/MUA/allOrderedMUA_ignore4secs_500HzBinning_2bins_' + monkey + '.npy', allow_pickle = True)
    # MUABINNED = np.load(basePath + monkey + '/MUA/allOrderedMUA_binsize_2_ignore4secs_' + monkey + '_15seconds.npy', allow_pickle = True)
    MUABINNED = np.load(basePath + monkey + '/MUA/allOrderedMUA_binsize_2_ignore4secs_' + monkey +  '_15seconds.npy', allow_pickle = True)
    # np.save(basePath + monkey + '/MUA/allOrderedMUA_ignore4secs_500HzBinning_2bins_' + monkey, MUABINNED)
    
    print(MUABINNED.shape)

    #% Creating array index, separating arrays, getting some useful index...
    num_anchors_perArray = 2

    lfp, raw_arrays, anchorIndices, colsL, allColors, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, orig_IDx = create_array_index_new(LFP[:,0:100],colsL,num_anchors_perArray, nonvalidE)
    
    del lfp, raw_arrays
    
    #anchor lfp channels that will be used for regression analysis
    anchorLFP = LFP[anchorIndices]
    anchorUtah = utah[anchorIndices]
    anchorUtahDistances = euclidean_distances(anchorUtah)
    # anchorUtahDistances /= utahEUmax # its important to divide them by the same number as the big Utah matrix, otherwise the regression
    
    utah = np.delete((utah),nonvalidE,axis=0)
    rfs = np.delete((rfs),nonvalidE,axis=0)
    LFP = np.delete((LFP),nonvalidE,axis=0)
    # LFP_gamma = np.delete((LFP_gamma),nonvalidE,axis=0)
    MUABINNED = np.delete((MUABINNED),nonvalidE,axis=0)
    
    colors = allColors
    arrayIDX = np.array(selectedChannelsIndices)
    
    #createAlphas for plotting
    colorsWAlpha = []
    for j in range(arrayIDX.shape[0]):
        
        singleArrIDX = arrayIDX[j]
        alphas = np.expand_dims(np.linspace(0.3,1,singleArrIDX.shape[0]),-1)
        alphas = np.power(alphas,2)
        arrayColorsWAlpha = np.hstack((allColors[singleArrIDX],alphas))
        # arrayColorsWAlpha = np.hstack((arrayIndexV1[singleArrIDX],alphas))
        colorsWAlpha.append(arrayColorsWAlpha)   
    
    colorsWAlpha = np.vstack(colorsWAlpha)


    return  LFP, MUABINNED, anchorLFP, anchorUtah, anchorUtahDistances, colors, arrayIDX, colorsWAlpha, \
        channelAreaMap, arrayNums, areas, channelNums, rfs_distances, utah, utah_distances, \
        badElectrodesDict, subjectConditionList, nonvalidEList, idxNonvalid, \
        nonvalidE, UTAHMAX,selectedChannelsIndices
        
def get_monkey_pixPerMM_utahMax(monkey):
    """
    Retrieve the pixels_per_mm and utahMax values based on the monkey identifier.

    Parameters:
        monkey (str): The identifier for the monkey.

    Returns:
        pixels_per_mm (float): The pixels_per_mm value for the given monkey.
        utahMax (float): The utahMax value for the given monkey.
    """
    if monkey == 'monkey_L':
        pixels_per_mm = 15.0
        utahMax = 1092 / 2
    elif monkey == 'monkey_A' or monkey == 'monkey_L':
        pixels_per_mm = 30.0
        utahMax = 1092
    else:
        print('    ERROR CHOOSING MONKEY')
        return None, None

    return pixels_per_mm, utahMax
        
def count_low_snr_electrodes(directory, monkey):
    # Get list of all csv files in the directory
    csv_files = glob.glob(f'{directory}/{monkey}/SNR/*.csv')
    
    # Initialize an empty dictionary to store the count of low SNR electrodes for each day
    low_snr_counts = {}
    
    # Loop through all csv files
    for file in csv_files:
        df = pd.read_csv(file)
        date = df['date'].iloc[0]
        low_snr_count = (df['SNR'] < 2).sum()
        
        # Store the low SNR count for the given date
        low_snr_counts[date] = low_snr_count

    return low_snr_counts



def loadMonkey3sessionsData_outliersOut_loadSpecificBand_15seconds(monkey, STANDARIZE_LFP, EXCLUDE_V4, basePath, whereUtahPath, channelAreaMappingPath, deleteElecsPath, freq_band="LFP", load_MUA=False):
    """
    This does the same as paper_figureV8_loading3sessions2monkeysdata.py but as a function
    
    basically it picks the utah, RFs, LFP and MUA data, colors etc. for that monkey
    
    This version of the function also loads ONLY the specified frequency band, and optionally it loads the MUABINNED data
    """
     
    colsL = [(1.0000, 0,  0,), (1.0000, 0.3750, 0), (1.0000, 0.7500, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706, 0.0745),
             (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0, 0.2500, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750)]
    colsA = [(1.0000, 0, 0), (1.0000, 0.3750, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0), (1.0000, 0.7500, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706,0.0745), 
             (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750), (0, 0, 0)]
    
    if monkey == 'monkey_L':
        None
    else:
        colsL = colsA
        
    # Creating colors for every electrode
    allColors =  None
    for i in range(len(colsL)):
    
        c = np.tile(colsL[i], (64,1))#,16)
 
        if i == 0:
            allColors = c
        else:
            allColors = np.vstack((allColors,c))
    
    # Loading channel to area map
    channelAreaMap = scipy.io.loadmat(channelAreaMappingPath)
    arrayNums = channelAreaMap['arrayNums']
    areas = channelAreaMap['areas']
    channelNums = channelAreaMap['channelNums']

    print('Loading RFs...')
    rfilePATH = os.path.join(basePath, monkey, 'RFS')
    if not os.path.exists(rfilePATH):
        rfilePATH = basePath + monkey + '/RFS/'
    rf_list = load_rf_instances(rfilePATH)
    
    # Order RFs according to their instances
    rfArrs, orderedRfs = order_RF_instances(rf_list, arrayNums, channelNums, V1ONLY=False)
    rfs = np.vstack(rfArrs)
    
    print('Now, while loading RFs, I am not calculating RF outliers, so nonvalidE = toDelete')
    print('TODO: Calculate RF outliers and merge them with toDelete array')
    
    # Load utah arrays
    # whereUtahPath = r'LFP/coordinates_of_electrodes_on_cortex_using_photos_of_arrays/'
    matName = 'allPixelIDs_monkey_L.mat' if monkey == 'monkey_L' else 'allPixelIDs_monkey_L.mat'
    utah = load_utah_XING(whereUtahPath, matName, allColors, PLOT=False)
    
    # print('NOT NORMALIZING RFS USING MAX BECAUSE MAX IS AN EXTREME VALUE')
    
    # Compute euclidean distances for RFs
    rfs_distances = euclidean_distances(rfs)
    
    # Translate and normalize utah electrode positions
    utah[:,1] += np.abs(utah[:,1].min())
    UTAHMAX = utah.max()
    utah /= UTAHMAX
    
    # Compute euclidean distances for utah
    utah_distances = euclidean_distances(utah)
    utah_distances /= utah_distances.max()
    
    
    # Compute 2 standard deviations for rfs_distances and utah_distances
    rfs_q1 = np.percentile(rfs_distances, 5)
    rfs_q3 = np.percentile(rfs_distances, 95)
    utah_std2 = 5 * np.std(utah_distances)
    
    # # Plotting RFs
    # plt.figure(dpi=300)
    # plt.scatter(rfs[:,0], rfs[:,1], c=allColors)
    # plt.ylim(-200, 100)
    # plt.xlim(-100, 200)
    
    # # Plotting RF and utah inter-distances
    # fig, axs = plt.subplots(2, dpi=180)
    
    # # Use 2 standard deviations for the colorbar range in imshow
    # img1 = axs[0].imshow(rfs_distances, cmap='afmhot', vmin=rfs_q1, vmax=rfs_q3)
    # axs[0].set_title('RFs electrode interdistances')
    # fig.colorbar(img1, ax=axs[0])
    
    # img2 = axs[1].imshow(utah_distances, cmap='afmhot', vmin=0, vmax=utah_std2)
    # axs[1].set_title('Utah electrode interdistances')
    # fig.colorbar(img2, ax=axs[1])
    
    # plt.tight_layout()
    
    arrayIndexV1 = allColors
    
    #% Loading Bad Channels
    ########### DELETING ELECTRODES IS NOW OVERRIDED BY LOADING NONVALID ELECTRODES LIST,
    ########### GENERATED IN paper_figuresV7_currentCopy.py
    ########### STORED IN 'LFP-RFs/data_results/deletedElectrodesDictionary/'
    
    if monkey == 'monkey_L':
        TOLOAD = 'monkey_LClosedEyes'
    else:
        TOLOAD = 'monkey_LClosedEyes'
        
    with open(os.path.join(deleteElecsPath, 'subjectCondition_NonvalidElectrodesList.pkl'), 'rb') as f:
        badElectrodesDict = pickle.load(f)
    subjectConditionList = np.array( badElectrodesDict['subject'] )
    nonvalidEList = badElectrodesDict['nonvalidElectrodesList']
    idxNonvalid = np.where( subjectConditionList == TOLOAD)[0][0]
    nonvalidE = nonvalidEList[idxNonvalid]
    
    #% LOADING ALL LFP AND MUA DATA (ignoring the first 4 seconds of each chunk of data)
    
    # EYES CLOSED
    # basePath + monkey + '/LFP/'
    # LFP = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_' + monkey + '_outlierChunks_removed.npy', allow_pickle=True)
    # print(LFP.shape)
    # basePath + monkey + '/MUA/'
    # # LFP_gamma = np.load(basePath + monkey + '/LFP/allOrderedLFP_ignore4secs_gamma_' + monkey + '.npy', allow_pickle=True)
    # MUABINNED = np.load(basePath + monkey + '/MUA/allOrderedMUA_binsize_2_ignore4secs_' + monkey + '_outlierChunks_removed.npy', allow_pickle = True)
    # print(MUABINNED.shape)


    # Conditionally load LFP data according to the specified frequency band
    # valid_freq_bands = ["LFP", "low","alpha", "beta", "gamma", "high_gamma"]
    valid_freq_bands = ["LFP", "low","alpha", "beta", "gamma", "highGamma"]
    if freq_band == 'high_gamma':
        freq_band = 'highGamma'
     
    # Mapping from given frequency bands to standardized names
    freq_band_mapping = {
    'original': 'LFP',
    'mattLow': 'low',
    'mattAlpha': 'alpha',
    'mattBeta': 'beta',
    'mattGamma': 'gamma',
    'mattHighGamma': 'highGamma'
    }

    freq_band = freq_band_mapping.get(freq_band, freq_band)     
    valid_freq_bands = ["LFP", "low", "alpha", "beta", "gamma", "highGamma"]

    if freq_band not in valid_freq_bands:
        raise ValueError(f"Invalid frequency band specified. Choose from {valid_freq_bands}")

    # LFP = np.load(basePath + monkey + f'/LFP/allOrdered{freq_band}_ignore4secs_' + monkey + '_outlierChunks_removed.npy', allow_pickle=True)
    
    if freq_band == 'LFP':
        LFP_path = os.path.join(basePath, monkey, 'LFP', f'allOrderedLFP_ignore4secs_{freq_band}_{monkey}_outlierChunks_removed_15seconds.npy')
    else:
        LFP_path = os.path.join(basePath, monkey, 'LFP', f'allOrderedLFP_ignore4secs_{freq_band}_{monkey}_outlierChunks_removed_15seconds.npy')
    
    LFP = np.load(LFP_path, allow_pickle=True)
    print(f"LFP path: {LFP_path}")

    # Conditionally load MUA data if specified
    MUABINNED = None  # Initialize to None
    if load_MUA:
        MUABINNED = np.load(os.path.join(basePath, monkey, 'MUA', f'allOrderedMUA_binsize_2_ignore4secs_MUA_{monkey}_outlierChunks_removed_15seconds.npy'), allow_pickle=True)
        print(MUABINNED.shape)

    #% Creating array index, separating arrays, getting some useful index...
    num_anchors_perArray = 2

    # using random array to obtain 1024 colors, deteling the bad ones later
    lfp, raw_arrays, anchorIndices, colsL, allColors, arrayNumberV1, colorSingleArray, relativeAnchorIdxList, selectedChannelsIndices, orig_IDx = create_array_index_new(np.random.rand(1024, 10),colsL,num_anchors_perArray, nonvalidE)
 
    del lfp, raw_arrays
    
    #anchor lfp channels that will be used for regression analysis
    anchorLFP = LFP[anchorIndices]
    anchorUtah = utah[anchorIndices]
    anchorUtahDistances = euclidean_distances(anchorUtah)
    
    utah = np.delete((utah),nonvalidE,axis=0)
    rfs = np.delete((rfs),nonvalidE,axis=0)
    
    # LFP AND MUABINNED WITH OUTLIER CHUNKS REMOVED ALREADY HAVE THE NONVALID ELECTRODES REMOVED 
    # LFP = np.delete((LFP),nonvalidE,axis=0)
    # LFP_gamma = np.delete((LFP_gamma),nonvalidE,axis=0)
    # MUABINNED = np.delete((MUABINNED),nonvalidE,axis=0)
    
    colors = allColors
    arrayIDX = selectedChannelsIndices
    
    #createAlphas for plotting
    colorsWAlpha = []
    for j in range(len(arrayIDX)):
        
        singleArrIDX = arrayIDX[j]
        alphas = np.expand_dims(np.linspace(0.3,1,len(singleArrIDX)),-1)
        alphas = np.power(alphas,2)
        arrayColorsWAlpha = np.hstack((allColors[singleArrIDX],alphas))
        # arrayColorsWAlpha = np.hstack((arrayIndexV1[singleArrIDX],alphas))
        colorsWAlpha.append(arrayColorsWAlpha)   
    
    colorsWAlpha = np.vstack(colorsWAlpha)

    return  LFP, MUABINNED, anchorLFP, anchorUtah, anchorUtahDistances, colors, arrayNumberV1, arrayIDX, colorsWAlpha, \
        channelAreaMap, arrayNums, areas, channelNums, rfs, rfs_distances, utah, utah_distances, \
        badElectrodesDict, subjectConditionList, nonvalidEList, idxNonvalid, \
        nonvalidE, UTAHMAX,selectedChannelsIndices
        
    
    
def map_and_normalize_day_colors(dayTags):
    # Initialize unique days from the provided dayTags
    unique_days = np.unique(dayTags)
    
    # Create a dictionary to map each unique day to a unique color
    color_dict = {}
    colorsD = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Red, Green, Blue
    for i, day in enumerate(unique_days):
        color_dict[day] = colorsD[i % 3]  # Cycle through colors if there are more than three days
    
    # Map each day in the original dayTags to its corresponding color
    color_days = np.asarray([color_dict[day] for day in dayTags], dtype='float32')
    
    # Normalize the colors
    color_days_normalized = color_days / 255.0
    
    # Filter for the unique normalized colors
    _, unique_indices = np.unique(color_days_normalized, return_index=True, axis=0)
    unique_colors = color_days_normalized[np.sort(unique_indices)]
    
    return unique_days, color_dict, color_days, color_days_normalized, unique_colors    



###############################################################################
###############################################################################
#
#
#           (\(\  
#            (-.-) > 
#            o_(")(")  
# =============================================================================
# 
# 
#      FREQUENCY ANALYSIS UTILS HERE 
# 
# 
# =============================================================================
###############################################################################
###############################################################################


def check_significance(data1, data2, alpha=0.05):
    """
    TODO Doc
    """
    t_stat, p_val = ttest_ind(data1, data2)
    return p_val < alpha


def load_most_recent_data(monkey, saveResultsPath):
    """
    TODO Doc
    """
    search_pattern = f"{saveResultsPath}{monkey}_v10_results_colors_utahMax_pixelsPerMm_*.npy"
    file_list = glob.glob(search_pattern)
    if not file_list:
        raise FileNotFoundError("No files matching the pattern found.")
    # Sort the files by modification time
    file_list.sort(key=os.path.getmtime, reverse=True)
    most_recent_file = file_list[0]
    loaded_data = np.load(most_recent_file, allow_pickle=True).item()
    results_by_method = loaded_data.get('results_by_method')
    allColors = loaded_data.get('allColors')
    utahMax = loaded_data.get('utahMax')
    pixels_per_mm = loaded_data.get('pixels_per_mm')  # Add this line
    colorsWAlpha = loaded_data.get('colorsWAlpha') 
    return results_by_method, allColors, colorsWAlpha, utahMax, pixels_per_mm


def plot_H(H,colors,title,alpha = None, dpi=180, subject = ''):
    """
    PLOTTING
    """
    fig = plt.figure(dpi=dpi)
    plt.scatter(H[:, 0], H[:, 1], s=50, cmap='Spectral',
                alpha=alpha,
                edgecolor="black",
                c=colors)  # c=arrayIndexV1,
    plt.title(title, size=15)
    plt.xlim(H.min(0)[0] -  abs(H.min(0)[0]) * 0.5, H.max(0)[0]* 1.05)
    if subject == 'monkey_LClosedEyes':
        # plt.xlim(H[:, 1].min() - abs(H[:, 1].min() * 1.3), H[:, 1].max()* 1.3)
        plt.xlim(0, H.max(0)[0]* 1.05)


def apply_dimension_reduction(method_name, X, n_neighbors=15, min_dist=0.1, corr_Ds=None, DIMENSIONS = 2):
    """
    Applies dimensionality reduction on data X based on the given method_name.
    Parameters:
    - method_name: str, the name of the method ('PCA', 'UMAP', 'MDS', 'PCACORR')
    - X: np.ndarray, the data matrix
    - n_neighbors: int, number of neighbors for UMAP (optional)
    - min_dist: float, minimum distance for UMAP (optional)
    - corr_Ds: function, function to calculate correlation matrix (only for 'MDS' and 'PCACORR')
    Returns:
    - H: np.ndarray, the transformed data matrix
    """        
    pca = PCA(n_components=DIMENSIONS)
    if method_name == 'PCA':
        ME = PCA(n_components=2)
    elif method_name == 'UMAP':
        pcaInit = PCA(n_components=2)
        init = pcaInit.fit_transform(X)
        ME = umap.UMAP(n_neighbors=n_neighbors, metric='euclidean', 
                       init=init, min_dist=min_dist, n_components=2)
    else:
        raise ValueError(f"Method {method_name} not recognized.")
    if method_name in ['MDS', 'PCACORR']:
        if corr_Ds is not None:
            print('Creating correlation matrix ')
            X = corr_Ds(X)[0]
        else:
            raise ValueError("corr_Ds function must be provided for 'MDS' and 'PCACORR'.")
    H = ME.fit_transform(X)
    return H


def apply_procrustes_and_evaluate(utah, H, utahMax, pixels_per_mm):
    """
    Applies Procrustes analysis, rescaling, and performance calculation on given utah and H matrices.
    Parameters:
    - utah: np.ndarray, the 'utah' data matrix
    - H: np.ndarray, the 'H' data matrix
    - utahMax: float, maximum value in 'utah'
    - pixels_per_mm: float, conversion factor for units
    Returns:
    - corr: float, the correlation of distances
    - rms: float, the root mean squared error
    - recovered_H: np.ndarray, the rescaled 'H' data matrix
    """
    utahPr, HPr, err, R, s, norm1, norm2, mean1, mean2 = scipy_Antonio_procrustes(utah, H)
    recovered_H = rescale_procrustes_map(H, R, s, norm1, norm2, mean1, mean2)
    corr = calculate_corr_of_distances(utah, recovered_H)
    rms = np.sqrt(mean_squared_error(utah / pixels_per_mm, 
                                     recovered_H / pixels_per_mm))
    return recovered_H, R, s, rms, corr


def evaluate_individual_arrays(utah, recovered_H, indi_arrayIdx, utahMax, pixels_per_mm):
    """
    Calculates performance metrics for individual arrays.
    Parameters:
    - utah: np.ndarray, the 'utah' data matrix
    - recovered_H: np.ndarray, the 'recovered_H' data matrix
    - indi_arrayIdx: np.ndarray, individual array indices
    - utahMax: float, maximum value in 'utah'
    - pixels_per_mm: float, conversion factor for units
    Returns:
    - corr_array_list: list of float, correlation values for each array
    - eu_array_list: list of float, RMS error values for each array
    """
    corr_array_list = []
    eu_array_list = []
    for array in range(indi_arrayIdx.shape[0]):
        singleArrIDX = indi_arrayIdx[array]
        utahSingle = utah[singleArrIDX]
        recoveredSingle = recovered_H[singleArrIDX]
        utahSingle -= utahSingle.mean(0)
        recoveredSingle -= recoveredSingle.mean(0)
        corr_array = calculate_corr_of_distances(utahSingle, recoveredSingle)
        rms_array = mean_squared_error(utahSingle * utahMax / pixels_per_mm, 
                                       recoveredSingle * utahMax / pixels_per_mm, 
                                       squared=False)
        corr_array_list.append(corr_array)
        eu_array_list.append(rms_array)
    return corr_array_list, eu_array_list


def compute_and_evaluate_dimension_reduction(seconds_per_chunk, method_name, band_name, X_data, utah, utahMax, indi_arrayIdx, pixels_per_mm, n_neighbors, min_dist, num_chunks):
    """
    Perform dimensionality reduction and evaluation for a given set of parameters and data.
    
    Parameters:
    - method_name (str): Name of the dimensionality reduction method (e.g., 'UMAP', 't-SNE')
    - band_name (str): Name of the frequency band under consideration (e.g., 'Alpha', 'Beta')
    - X_data (numpy array): Data to be reduced (n_channels x n_timepoints)
    - utah (numpy array): Utah array coordinates
    - utahMax (float): Maximum distance between Utah array electrodes
    - indi_arrayIdx (numpy array): Indices of individual arrays in the dataset
    - pixels_per_mm (float): Conversion factor for dimensions
    - n_neighbors (int): Number of neighbors for dimensionality reduction method
    - min_dist (float): Minimum distance parameter for dimensionality reduction method
    
    Returns:
    - results (dict): Dictionary containing correlations, errors, and other evaluation metrics
    """
    
    results = {'correlations': None, 'errors': None, 'maps': None, 'corr_array_list': None, 'eu_array_list': None}
    
    # Create a new random number generator for this thread
    local_rng = default_rng()
    
    # Randomly choose a time chunk
    chunk = local_rng.choice(num_chunks, size=1, replace=True)[0]
    idx1 = chunk * seconds_per_chunk * 500
    idx2 = (chunk + 1) * seconds_per_chunk * 500
    X = X_data[:, idx1: idx2]

    try:
        H = apply_dimension_reduction(method_name, X, n_neighbors, min_dist)
    except Exception as e:
        print('in apply_dimension_reduction ', flush=True)
        print(e, flush=True)

    try:
        recovered_H, _, _, rms, corr = apply_procrustes_and_evaluate(utah, H, utahMax, pixels_per_mm)
    except Exception as e:
        print('in apply_procrustes_and_evaluate ', flush=True)
        print(e, flush=True)

    try:
        corr_array_list, eu_array_list = evaluate_individual_arrays(utah, recovered_H, indi_arrayIdx, utahMax, pixels_per_mm)
    except Exception as e:
        print('in evaluate_individual_arrays ', flush=True)
        print(e, flush=True)

    results['correlations'] = corr
    results['errors'] = rms
    results['maps'] = recovered_H
    results['corr_array_list'] = corr_array_list
    results['eu_array_list'] = eu_array_list
    return results


def plot_median_maps(results_by_method, bandNames, methodNames, utahMax, pixels_per_mm, saveResultsPath, colorsWAlpha, monkey, SAVEFIG = False):
    """
    Plot median maps of neural data for various frequency bands and methods.
    Parameters:
    - results_by_method (dict): Nested dictionary containing neural map data organized by method and frequency band.
    - bandNames (list): List of frequency band names.
    - methodNames (list): List of neural processing method names.
    - colorsWAlpha (ndarray): Colors to be used for the scatter plot, with alpha channels.
    - monkey (str): Identifier for the specific monkey being analyzed.
    - SAVEFIG (bool, optional): Flag indicating whether to save the figure. Default is False.

    No returns. Plots are displayed and optionally saved.
    """
    for method in methodNames:
        for band in bandNames:
            # Create a new figure for each method and band
            fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Extract 'maps' data and calculate median
            maps_data = np.array(results_by_method[method][band]['maps'])
            median_map = np.median(maps_data, axis=0)
            x = median_map[:, 0] * utahMax / pixels_per_mm
            y = median_map[:, 1] * utahMax / pixels_per_mm
            
            ax.scatter(x, y, c=colorsWAlpha, edgecolor = 'black')
            plt.ylim(-0.1 * utahMax / pixels_per_mm, 0.7 * utahMax / pixels_per_mm)
            plt.xlim(0.2 * utahMax / pixels_per_mm, 1.0 * utahMax / pixels_per_mm)
            plt.xlabel("X Coordinate of Median Map", fontsize=14)
            plt.ylabel("Y Coordinate of Median Map", fontsize=14)
            plt.title(f"Median of Maps ({method}, {band})", fontsize=16)
            if SAVEFIG:
                print('saving figures ')
                print(f"{monkey}_Median_of_Maps_{method}_{band}")
                plt.savefig(f"{saveResultsPath}{monkey}_Median_of_Maps_{method}_{band}.tif", format='tif', dpi=300)
                plt.savefig(f"{saveResultsPath}{monkey}_Median_of_Maps_{method}_{band}.svg", format='svg', dpi=300)
                plt.savefig(f"{saveResultsPath}{monkey}_Median_of_Maps_{method}_{band}.pdf", format='pdf', dpi=300)
            plt.show()
  

def plot_metric_perFreq_perMethod(metrics_to_plot, num_bands, methodNames, bandNames, results_by_method, method_color_map, saveResultsPath, plot_type='scatter', monkey="", SAVEFIG=False):
    """
    Plot specified metrics across frequency bands and methods.
    Parameters:
    - metrics_to_plot (list): Metrics to be plotted, e.g. ['correlations', 'RMSE'].
    - plot_type (str, optional): Type of plot ('scatter' or 'violin'). Default is 'scatter'.
    - monkey (str, optional): Identifier for the specific monkey being analyzed. Default is an empty string.
    - SAVEFIG (bool, optional): Flag indicating whether to save the figure. Default is False.
    No returns. Plots are displayed and optionally saved.
    Note:
    The function uses global variables such as `methodNames`, `bandNames`, `results_by_method`, `method_color_map`, 
    and `saveResultsPath`. These need to be defined before calling the function.
    """
    
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(dpi=300)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)  
        tick_spacing = 4
        tick_positions = np.arange(0, num_bands * tick_spacing, tick_spacing)
        mean_values = {method: [] for method in methodNames}
        data_values = {method: {band: [] for band in bandNames} for method in methodNames}
        max_data_value = 0  # To calculate a common y-coordinate for asterisks
        
        for i, method in enumerate(methodNames):
            for j, data_type in enumerate(results_by_method[method].keys()):
                data = np.array(results_by_method[method][data_type][metric])
                jitter = np.random.normal(0, 0.15, len(data))
                method_color = method_color_map.get(method, 'gray')
                
                if plot_type == 'scatter':
                    # Scatterplot
                    ax.scatter(tick_positions[j] + jitter, data, alpha=0.3, edgecolors='none', s=30, color=method_color)
                    bp = ax.boxplot(data, positions=[tick_positions[j]], widths=0.6, notch=True, patch_artist=True)
                    # Customizing the boxplot: Setting facecolor to none makes it transparent
                    for box in bp['boxes']:
                        box.set(facecolor='none', edgecolor='black')
        
                elif plot_type == 'violin':
                    # Violin plot
                    sns.violinplot(x=tick_positions[j] + jitter, y=data, ax=ax, color=method_color, alpha=0.5)
                mean_values[method].append(np.mean(data))
                data_values[method][data_type] = data  # Store data for significance check  
                max_data_value = max(max_data_value, max(data))  # Update max data value

        # Set Y-axis limits for correlations
        if 'cor' in metric:            
            # plt.ylim(0, max_data_value + 0.1)
            plt.ylim(0, 1)
        
        for band in bandNames:
            # Check for significance
            if len(methodNames) >= 2:
                method1, method2 = methodNames[:2]  # Assuming you want to compare the first two methods
                is_significant = check_significance(data_values[method1][band], data_values[method2][band])
                if is_significant:
                    # Calculate a common y-coordinate for asterisks
                    y_coord = max_data_value - max_data_value * 0.05
                    # Add an asterisk for significance
                    ax.text(tick_positions[bandNames.index(band)], y_coord, '*', fontsize=24)
        
        for method, means in mean_values.items():
            ax.plot(tick_positions, means, '-o', label=f"{method} mean {metric}", color=method_color_map.get(method, 'gray'))
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(bandNames, size=10)
        plt.xlabel('Frequency Band', size=14)
        plt.ylabel(f'{metric} Value', size=14)
        plt.title(f'Metric Value vs Frequency Band ({metric})', size=16)

        if SAVEFIG:
            print('saving figures')
            print(f"{monkey}_Metric_vs_Frequency_{metric}")
            plt.savefig(f"{saveResultsPath}{monkey}_Metric_vs_Frequency_{metric}.tif", format='tif', dpi=300)
            plt.savefig(f"{saveResultsPath}{monkey}_Metric_vs_Frequency_{metric}.svg", format='svg', dpi=300)
            plt.savefig(f"{saveResultsPath}{monkey}_Metric_vs_Frequency_{metric}.pdf", format='pdf', dpi=300)
        plt.show()     


def plot_metric_perFreq_perMethod_SINGLEARRAY(metrics_to_plot, num_bands, methodNames, bandNames, results_by_method, method_color_map, saveResultsPath, plot_type='scatter', monkey="", SAVEFIG=False):
    """
    Generate plots for specific metrics across frequency bands for each method.
    This function uses a single 2D array for plotting and adds boxplots on top of scatter plots.

    Parameters
    ----------
    metrics_to_plot : list
        List of metrics to be plotted. E.g., ['correlations', 'RMSE'].
    plot_type : str, optional
        Type of plot to display. Default is 'scatter'.
    monkey : str, optional
        Identifier for the specific monkey being analyzed. Default is an empty string.
    SAVEFIG : bool, optional
        Flag indicating whether to save the figure or not. Default is False.

    Returns
    -------
    None
        No returns. Plots are displayed and optionally saved as specified by the SAVEFIG flag.

    Notes
    -----
    The function relies on several global variables including:
    - `methodNames`: Names of the methods to be analyzed.
    - `bandNames`: Names of the frequency bands.
    - `results_by_method`: Data structure containing the metric values.
    - `method_color_map`: Dictionary mapping methods to colors.
    - `saveResultsPath`: Path where the results are to be saved.

    Examples
    --------
    >>> plot_metric_perFreq_perMethod_SINGLEARRAY(['correlations'], plot_type='scatter', monkey='M1', SAVEFIG=True)

    """
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(dpi=300)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        tick_positions = np.arange(0, len(bandNames) * 4, 4)
        mean_values = {method: [] for method in methodNames}
        max_y_value = 0  # to set the y-coordinate for the asterisk
        
        for i, method in enumerate(methodNames):
            for j, band in enumerate(bandNames):
                data = np.array(results_by_method[method][band][metric])
                data = data.mean(1)
                
                jitter = np.random.normal(0, 0.15, len(data))
                
                if plot_type == 'scatter':
                    ax.scatter(tick_positions[j] + jitter, data, alpha=0.3, edgecolors='none', s=30, color=method_color_map.get(method, 'gray'))
                
                # boxplot
                bp = ax.boxplot(data, positions=[tick_positions[j]], widths=0.6, notch=True, patch_artist=True)
                
                for box in bp['boxes']:
                    box.set(facecolor='none', edgecolor='black')
                    
                for whisker in bp['whiskers']:
                    whisker.set(color='black')
                for cap in bp['caps']:
                    cap.set(color='black')
                for median in bp['medians']:
                    median.set(color='black')
                    
                    
                mean_values[method].append(data.mean())
                max_y_value = max(max_y_value, data.max())
        
        for method, means in mean_values.items():
            ax.plot(tick_positions, means, '-o', label=f"{method} mean {metric}", color=method_color_map.get(method, 'gray'))
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(bandNames, size=10)
        plt.xlabel('Frequency Band', size=14)
        plt.ylabel(f'{metric} Value', size=14)
        plt.title(f'Metric Value vs Frequency Band ({metric})', size=16)

        if SAVEFIG:
            print('saving figures')
            print(f"{monkey}_Metric_vs_Frequency_{metric}")
            plt.savefig(f"{saveResultsPath}{monkey}_Metric_vs_Frequency_{metric}.tif", format='tif', dpi=300)
            plt.savefig(f"{saveResultsPath}{monkey}_Metric_vs_Frequency_{metric}.svg", format='svg', dpi=300)
            plt.savefig(f"{saveResultsPath}{monkey}_Metric_vs_Frequency_{metric}.pdf", format='pdf', dpi=300)
        plt.show()   
