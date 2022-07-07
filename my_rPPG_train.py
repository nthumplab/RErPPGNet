from __future__ import print_function, division
import numpy as np
import cv2
from  tqdm import tqdm
import os
import torch
import torch.nn as nn
import dataloader
from new_my_rPPG_model import NegPearsonLoss, Physnet


def shuffle_tensor(batch_tensor):
    batch_size = batch_tensor.shape[0]
    shuffled_batch_tensor = batch_tensor
    while True:
        id = torch.randperm(batch_size)
        shuffled_batch_tensor = shuffled_batch_tensor[id]
        for i in range(batch_size):
            if torch.all(torch.eq(shuffled_batch_tensor[i],batch_tensor[i])):
                break
        else:
            return shuffled_batch_tensor
class VarianceLoss(nn.Module):
    def __init__(self):
        super(VarianceLoss, self).__init__()
        return

    def forward(self, x):
        # for i in range(x.shape[0]):
        vx = x - torch.mean(x, dim = 2, keepdim = True)
        r = torch.sum(vx * vx) / x.shape[2]
        return r

def run():
    att_type = 'MyCBAM_v3' # without/ old/ CBAM/ MyCBAM/ MyCBAM_v2/ MyCBAM_v3
    #use_pretrain = False
    load_opt = True
    network_name = f"rPPG_with_{att_type}_attention"
    # Step 1: Set file name
    rPPG_weight_folder_name = network_name + "_weight"
    os.makedirs(rPPG_weight_folder_name, exist_ok=True)

    # Step2: Step train args
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_batch_size = 3 # default 2
    seq_length = 60
    lr = 0.001

    Physnet_rppg = Physnet(seq_length = seq_length, att_type=att_type).to(device0)
    optimizer_real_rPPG = torch.optim.Adam(Physnet_rppg.parameters(), lr=lr)
    #print(Physnet_rppg)
    
    real_rppg_criterion = NegPearsonLoss().to(device0)

    # Step 3: Load pretrained model
    epoch_rPPG = 0
    print(f"{rPPG_weight_folder_name}")
    try:
        latest_pkl_path = sorted(os.listdir(f'./{rPPG_weight_folder_name}'))[-1]
        checkpoint = torch.load(f'./{rPPG_weight_folder_name}/{latest_pkl_path}', map_location=device0)
        Physnet_rppg.load_state_dict(checkpoint['model_state_dict'])
        optimizer_real_rPPG.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_rPPG = checkpoint['epoch']
        print(f'load the {epoch_rPPG}th epoch from real_rppg')
    except:
        print('no model found for real_rppg')
        optimizer_real_rPPG = torch.optim.Adam(Physnet_rppg.parameters(), lr=lr)
        epoch_rPPG = 0

    trainloader = dataloader.load_pure_train(batch_size=train_batch_size, seq_length=seq_length)
    # Step 4: Start Train
    while epoch_rPPG <= 500:
        epoch_rPPG += 1
        print("epoch_rppg:", epoch_rPPG)

        train_running_real_rppg_loss = 0.0
        train_running_mid_real_rppg_loss = 0.0
        for batch, sample_batched in tqdm(enumerate(trainloader)):
            print(f'batch:{batch}/{len(trainloader)}')
            Physnet_rppg.train()
            optimizer_real_rPPG.zero_grad()

            ###### using real rppg to train ########
            real_label_rppg = sample_batched['label'].to(device0)
            real_rppg, mid_real_rppg,  map1 = Physnet_rppg(sample_batched['face_frames'].to(device0))
            real_rppg = real_rppg[:,0,:,0,0]
            mid_real_rppg = mid_real_rppg[:,0,:,0,0]

            real_rppg_loss = real_rppg_criterion(real_rppg, real_label_rppg)
            mid_real_rppg_loss = real_rppg_criterion(mid_real_rppg, real_label_rppg)

            loss = real_rppg_loss + mid_real_rppg_loss

            loss.backward()
            optimizer_real_rPPG.step()
            train_running_real_rppg_loss += real_rppg_loss.item()
            train_running_mid_real_rppg_loss += mid_real_rppg_loss.item()

            print('real_rppg_loss:', real_rppg_loss.item())
            print('mid_real_rppg_loss:', mid_real_rppg_loss.item())

        torch.save({
            'epoch': epoch_rPPG,
            'model_state_dict': Physnet_rppg.state_dict(),
            'optimizer_state_dict': optimizer_real_rPPG.state_dict()
        }, f'./{rPPG_weight_folder_name}/{"{:0>4d}".format(epoch_rPPG)}_{"{:0>4d}".format(int(len(trainloader) / train_batch_size))}.pkl')

        with open(f'{network_name}_training_loss.txt', 'a') as f:
            f.write(f'epoch_rppg:{epoch_rPPG}\n')
            f.write(f' real_rppg_loss:{"{0:.4f}".format(train_running_real_rppg_loss / (len(trainloader)))}\n')
            f.write(f' mid_real_rppg_loss:{"{0:.4f}".format(train_running_mid_real_rppg_loss / (len(trainloader)))}\n')

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    run()
