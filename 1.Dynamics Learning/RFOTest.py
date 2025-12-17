
import copy

import tensorflow as tf

import RtrainingTensor2x

import numpy as np

import os
tf.compat.v1.disable_eager_execution()

params = {}
input_dim = 6;
out1 = 15;

def count_subdirectories(path):
    try:
        subdirectories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        return len(subdirectories)
    except FileNotFoundError:
        print(f"Path '{path}' dose not exist")
        return 0
    except NotADirectoryError:
        print(f"Path '{path}' is not a directory")
        return 0





def PhyInput(X, r):
    """
    This aims to generate the hidden layer value for a PhyTaylor NN
    Arguments:
        X -- input array of shape (R, Q)
        r -- integer representing the expansion order
    Returns:
        MXR -- array with the hidden layer values
    """

    # a.Suppress Input
    N = X.shape[0]
    Q = X.shape[1]

    # Initialize the output list
    MXR = []

    # b. Generate Hidden Nodes
    for i in range(Q):
        mxr = X[:, i].reshape(-1, 1)
        tempmxr = X[:, i].reshape(-1, 1)
        mcounting = np.arange(N) + 1

        for j in range(2, r + 1):
            tb_list = []
            for z in range(N):
                ta = X[z, i] * tempmxr[mcounting[z] - 1:]
                if z == 0:
                    tb = ta
                else:
                    tb = np.vstack((tb, ta))
                mcounting[z] = tb.shape[0] - ta.shape[0] + 1
            tempmxr = tb
            mxr = np.vstack((mxr, tb))

        if len(MXR) == 0:
            MXR = mxr
        else:
            MXR = np.hstack((MXR, mxr))

    # Add the row of ones
   # ones_row = np.ones((1, Q))
    #MXR = np.vstack((ones_row, MXR))

    return MXR


# Example usage:
expansionrank=3;
##sampling period
T = 0.005;


params['data_name'] = 'Car_test'
params['seed'] = 9
params['uncheckable_dist_weights'] = ['tn','tn']
params['uncheckable_output_size'] = [input_dim,out1,6]
params['uncheckable_epd'] = np.array([expansionrank-1,0])
params['uncheckable_act'] = ['elu','none']
params['uncheckable_com_type1'] = ['none','none']
params['uncheckable_com_type2'] = ['none','none']
params['uncheckable_dist_biases'] = ['normal','normal']
params['uncheckable_num_of_layers'] = len(np.array([0,0])) 

##embed physical knowledge#################################################################################################





##
Phy_lay1_B_Initial=np.zeros((6,1), dtype=np.float32);
Phy_lay1_B_Initial[[0,3],0]=1;
Phy_lay1_B_line4=PhyInput(Phy_lay1_B_Initial,expansionrank)
##layer 1##
a_input_dim = Phy_lay1_B_line4.shape[0];

Phy_lay1_A = np.zeros((out1,a_input_dim), dtype=np.float32)
Phy_lay1_B = np.zeros((out1,a_input_dim), dtype=np.float32)
phyBias_lay1_A = np.zeros((out1,1), dtype=np.float32)
phyBias_lay1_B = np.ones((out1,1), dtype=np.float32)

Phy_lay1_A[0][0] = 1;
Phy_lay1_A[0][3] = T;
Phy_lay1_A[1][1] = 1;
Phy_lay1_A[1][4] = T;
Phy_lay1_A[2][2] = 1;
Phy_lay1_A[2][5] = T;
Phy_lay1_A[3][3] = 1;
Phy_lay1_A[4][4] = 1;
Phy_lay1_A[5][5] = 1;



Phy_lay1_B[3,:]=Phy_lay1_B_line4.T;
Phy_lay1_B[4:]=0
phyBias_lay1_B[0:3] = 0
###########

##Layer2##
Phy_lay2_A = np.zeros((6,out1), dtype=np.float32)
Phy_lay2_B = np.ones((6,out1), dtype=np.float32)
phyBias_lay2_A = np.zeros((6,1), dtype=np.float32)
phyBias_lay2_B = np.ones((6,1), dtype=np.float32)

Phy_lay2_A[0][0] = 1;
Phy_lay2_A[1][1] = 1;
Phy_lay2_A[2][2] = 1;
Phy_lay2_A[3][3] = 1;

Phy_lay2_B[0] = 0;
Phy_lay2_B[1] = 0;
Phy_lay2_B[2] = 0;
Phy_lay2_B[3] = 0;

phyBias_lay2_B[0:3] = 0
###########

params['uncheckable_phyweightsA'] = [Phy_lay1_A, Phy_lay2_A]
params['uncheckable_phyweightsB'] = [Phy_lay1_B, Phy_lay2_B]
params['uncheckable_phybiasesA'] = [phyBias_lay1_A, phyBias_lay2_A]
params['uncheckable_phybiasesB'] = [phyBias_lay1_B, phyBias_lay2_B]
###########################################################################################################################

params['traj_len'] = 200   #must equate to batch size
params['Xwidth'] = 6
params['Ywidth'] = 6
params['lYwidth'] = 6
params['dynamics_lam'] = 1;


params['exp_name'] = 'exp'
params['folder_name'] = 'NHPTaylor'


params['number_of_data files_for_training'] = 4
#params['num_passes_per_file'] =  1000 * 8 * 1000
params['num_passes_per_file'] = 100
params['batch_size'] = 200
params['num_steps_per_batch'] = 2
params['loops for val'] = 1
directory_path = 'subNetworkFolders'


for count in range(1,2):
#for count in range(1,count_subdirectories(directory_path)+1):
     params['model_path'] = "subNetworkFolders/SubNetworkConfiguration%d/%s_model.ckpt" % (count,params['exp_name'])
     params['subnn_path'] = "subNetworkFolders/SubNetworkConfiguration%d" %count
     params['subnn_configuration']="%s/IntervalValue.csv" % params['subnn_path']
     print(f"Starting TrainingError for SubNN {count}",flush=True)
     RtrainingTensor2x.main_exp(copy.deepcopy(params))
print('Done Training for subNN_Number %d' %count,flush=True)