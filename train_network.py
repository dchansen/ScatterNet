import SimpleITK as sitk
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

torch.backends.cudnn.benchmark = True


def load_mha(mha_file):
    itk = sitk.ReadImage(mha_file)
    return sitk.GetArrayFromImage(itk)


def load_projections(files):
    projections = [load_mha(f) for f in files]
    projections = np.concatenate(projections, 0)
    projections = np.pad(projections, [(0, 0), (4, 4), (4, 4)], mode="edge")
    projections[projections < 0] = 0
    return projections[:, np.newaxis, ...]


class ProjectionDatasSet(TensorDataset):
    # Mixup
    def __init__(self, data_array, target_array, distribution=np.random.rand):
        super(ProjectionDatasSet, self).__init__(data_array, target_array)
        self.distribution = distribution

    def __getitem__(self, item):
        data, target = super(ProjectionDatasSet, self).__getitem__(item)

        other_item = np.random.randint(0, self.__len__())

        mix = self.distribution()
        data2, target2 = super(ProjectionDatasSet, self).__getitem__(other_item)

        data_mixed = data * mix + data2 * (1 - mix)
        target_mixed = target * mix + target2 * (1 - mix)

        return data_mixed, target_mixed


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_Conv_type(m):
    return isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv2d) or \
           isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or \
           isinstance(m, nn.Linear)


def InitModel(model):
    for m in model.modules():
        if is_Conv_type(m):
            nn.init.orthogonal(m.weight.data)


import time
import tensorboardX

writer = tensorboardX.SummaryWriter()

use_cuda = True


def train_model(model, optimizer, dset_loaders, num_epochs=200, scheduler=None, start_epoch=0, criterion=nn.MSELoss()):
    since = time.time()
    batch_time = AverageMeter()
    running_loss = {"val": AverageMeter(), "train": AverageMeter()}
    best_model = model
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        # optimizer.update_step()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        k = 0

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_corrects = 0
            i = 0
            report = 200

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, targets = data

                # wrap them in Variable
                if use_cuda:
                    inputs, targets = Variable(inputs.float().cuda(async=True)), Variable(
                        targets.float().cuda(async=True))
                else:
                    inputs, targets = Variable(inputs.float()), Variable(
                        targets.float())

                if phase == "val":
                    inputs.volatile = True
                    targets.volatile = True
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                loss = criterion(outputs, targets)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                if isinstance(criterion, nn.MSELoss):
                    base_loss = loss
                else:
                    base_loss = nn.MSELoss()(torch.log(outputs), torch.log(targets))
                    print("Penguins were here")
                print("Loss", base_loss.data[0], "Time",
                      time.time() - since)
                batch_time.update(time.time() - since)
                since = time.time()
                # statistics

                running_loss[phase].update(base_loss.data[0], n=outputs.size()[0])
                i += 1

            writer.add_scalar('Loss_' + phase, running_loss[phase].avg, epoch)
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step(running_loss[phase].avg)

            batch_time.reset()

        for phase in ["val", "train"]:
            running_loss[phase].reset()

        torch.save(model.state_dict(), writer.file_writer.get_logdir() + "/model_" + str(epoch) + ".trch")

        print()

    writer.close()
    return model


training_patients = [2, 3, 4, 5, 7, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29]

projection_files = ["NewProjections/CBCTcor" + str(k) + "/ProjectionData/CBCT_projections_rtk_binned.mha" for k in
                    training_patients]
corprojection_files = ["NewProjections/CBCTcor" + str(k) + "/ProjectionData/CBCT_projections_cor_CF_1.6.mha" for k in
                       training_patients]

test_patients = [8, 9, 10, 12, 13, 14, 15]

test_projection_files = ["NewProjections/CBCTcor" + str(k) + "/ProjectionData/CBCT_projections_rtk_binned.mha" for k in
                         test_patients]
test_corprojection_files = ["NewProjections/CBCTcor" + str(k) + "/ProjectionData/CBCT_projections_cor_CF_1.6.mha" for k
                            in test_patients]

distribution = np.random.rand

train_loader = DataLoader(TensorDataset(torch.from_numpy(load_projections(projection_files)),
                                        torch.from_numpy(load_projections(corprojection_files))), batch_size=8,
                          shuffle=True, pin_memory=True)
test_loader = DataLoader(TensorDataset(torch.from_numpy(load_projections(test_projection_files)),
                                       torch.from_numpy(load_projections(test_corprojection_files))), batch_size=8,
                         shuffle=False, pin_memory=True)

import ScatterNet

model = ScatterNet.ScatterNet(init_channels=8, layer_channels=[8, 16, 32, 64, 128, 256], batchnorm=False, squeeze=False,
                              activation=nn.PReLU, exp=False,
                              skip_first=False, residual=True)

InitModel(model)

torch.save(model, writer.file_writer.get_logdir() + "/base_model.trch")

dummy_data = None

if use_cuda:
    model = nn.DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0)

train_model(model, optimizer, {"val": test_loader, "train": train_loader}, num_epochs=10000)
