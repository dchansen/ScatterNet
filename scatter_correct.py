import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torch.utils.data  import DataLoader,TensorDataset
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("files",nargs='+',help='Projection files to scatter correct')
parser.add_argument('--output_dir',help="Output directory")

args = parser.parse_args()

import ScatterNet

model = ScatterNet.ScatterNet(init_channels=8,layer_channels=[8,16,32,64,128,256],batchnorm=False,squeeze=False,
                  activation=nn.PReLU,exp=False,
                  skip_first=False,residual=True)


state_dict = torch.load("trained_models/model_103.trch") #Model used for article
model = nn.DataParallel(model)
model.load_state_dict(state_dict)

for proj_file in args.files:
    print(dir)
    stk_projections = sitk.ReadImage(proj_file)



    data = sitk.GetArrayFromImage(stk_projections)

    print("Loaded")
    data = np.pad(data,[(0,0),(4,4),(4,4)],mode="edge")

    print("Padded")



    loader= DataLoader(TensorDataset(torch.from_numpy(data[:,np.newaxis,...]),torch.from_numpy(data[:,np.newaxis,...])),batch_size=8,pin_memory=True)

    total_projections = []
    for projections,_ in loader:
        with torch.no_grad():
            var = Variable(projections.float())
            data_net_corrected = model(var)

        data_net_corrected = data_net_corrected.data.cpu().numpy()
        total_projections.append(data_net_corrected)

    total_projections = np.concatenate(total_projections,axis=0)[:,0,...]
    total_projections[np.isinf(total_projections)] = 0
    total_projections = total_projections[:,4:-4,4:-4]



    total_projections = sitk.GetImageFromArray(total_projections)
    total_projections.CopyInformation(stk_projections)
    sitk.WriteImage(total_projections,args.output_dir + "/" + os.path.basename(proj_file))





