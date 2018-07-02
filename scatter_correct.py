import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torch.utils.data  import DataLoader,TensorDataset
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import glob


dirs = glob.glob("NewProjections/CBCTcor*")


import ScatterNet

model = ScatterNet.ScatterNet(init_channels=8,layer_channels=[8,16,32,64,128,256],batchnorm=False,squeeze=False,
                  activation=nn.PReLU,exp=False,
                  skip_first=False,residual=True)


state_dict = torch.load("trained_models/model_103.trch") #Model used for article
model = nn.DataParallel(model)
model.load_state_dict(state_dict)

for dir in dirs:
    print(dir)
    projections = sitk.ReadImage(dir+"/ProjectionData/CBCT_projections_rtk_binned.mha")
    corrected_projections = sitk.ReadImage(dir+"/ProjectionData/CBCT_projections_cor_CF_1.6.mha")
    # corrected_projections = projections


    data = sitk.GetArrayFromImage(projections)
    data_corr = sitk.GetArrayFromImage(corrected_projections)
    print("Loaded")
    data = np.pad(data,[(0,0),(4,4),(4,4)],mode="edge")

    print("Padded")


# var = Variable(torch.from_numpy(data[:,np.newaxis,...]).float().cuda())


    loader= DataLoader(TensorDataset(torch.from_numpy(data[:,np.newaxis,...]),torch.from_numpy(data_corr[:,np.newaxis,...])),batch_size=8,pin_memory=True)

    total_projections = []
    for projections,_ in loader:
        var = Variable(projections.float())
        var.volatile = True
        data_net_corrected = model(var)
        # data_net_corrected = -torch.log(data_net_corrected/65536)
        data_net_corrected = data_net_corrected.data.cpu().numpy()
        total_projections.append(data_net_corrected)

    total_projections = np.concatenate(total_projections,axis=0)[:,0,...]
    total_projections[np.isinf(total_projections)] = 0
    total_projections = total_projections[:,4:-4,4:-4]

    print(dir, np.mean((total_projections-data_corr)**2))

    total_projections = sitk.GetImageFromArray(total_projections)
    total_projections.CopyInformation(corrected_projections)
    sitk.WriteImage(total_projections,dir+"/ProjectionData/ScatterNet2_projections.mha")





