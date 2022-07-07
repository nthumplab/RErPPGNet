from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
import os
import dataloader
from my_rPPG_model import NegPearsonLoss, Physnet
from embed_rPPG_att_model import RppgAtt_v2
from opts import get_args, get_result_folder_name
from utils import *
from random import sample

rppg_net_att_type = 'MyCBAM_v3'
args = get_args()

att_type = 'MyCBAM_v3'
network_name = f"rPPG_with_{att_type}_attention"
real_rppg_weight_folder_name = network_name + "_weight"

result_folder_name = get_result_folder_name(args)

de_rppg_weight_folder_name = os.path.join(result_folder_name, "weight_de")
embed_rppg_weight_folder_name = os.path.join(result_folder_name, "weight_embed")

device_rppg = torch.device(args.device_rppg if torch.cuda.is_available() else "cpu")
device_syn = torch.device(args.device_syn if torch.cuda.is_available() else "cpu")
print(f"device_rppg: {device_rppg}/ device_syn:{device_syn}")
train_batch_size = 1
seq_length = args.syn_seq_length # 100
test_per_epoch = 1
lr = 0.001

consistency_weight = 100.0
reconstruction_weight = 100.0
feature_consistency_weight = 10.0
rppg_weight = 1.0
rppg_rampup = 100


os.makedirs(de_rppg_weight_folder_name, exist_ok=True)
os.makedirs(embed_rppg_weight_folder_name, exist_ok=True)


class VarianceLoss(nn.Module):
    def __init__(self):
        super(VarianceLoss, self).__init__()
        return

    def forward(self, x):
        # for i in range(x.shape[0]):
        vx = x - torch.mean(x, dim = 2, keepdim = True)
        r = torch.sum(vx * vx) / x.shape[2]
        return r

def get_fake_rppg(PPG_set):
    fake_set = sample(PPG_set,1)[0]
    index_set = [x for x in range(len(fake_set)-seq_length)]
    index = sample(index_set,1)[0]
    output = fake_set[index:index+seq_length]
    return torch.FloatTensor([output])


fake_label = dataloader.get_pure_fake_label('/shared/pure', 'train01.txt')
fake_label_set = []
for fake_ppg in fake_label:
    fake_label_set.append(fake_ppg)
print(len(fake_label_set))

trainloader = dataloader.load_pure_train(batch_size=train_batch_size, seq_length=seq_length)
print("trainloader done")

Physnet_rppg = Physnet(seq_length = seq_length, att_type = rppg_net_att_type).to(device_rppg)
print("physnet done")

De_rPPG_encoder = RppgAtt_v2(medium_channels = 8, task="removal").to(device_syn)
Rppg_embeder = RppgAtt_v2(medium_channels = 8, task="embedd").to(device_syn)
print("Synthesis network done")

optimizer_real_rPPG = torch.optim.Adam(Physnet_rppg.parameters(), lr=lr)
optimizer_de_rPPG = torch.optim.Adam(De_rPPG_encoder.parameters(), lr=lr)
optimizer_embed_rPPG = torch.optim.Adam(Rppg_embeder.parameters(), lr=lr)
print("optimizer done")


removal_consistency_criterion = nn.L1Loss().to(device_syn)
embedding_consistency_criterion = nn.L1Loss().to(device_syn)
removal_reconstruction_criterion = nn.L1Loss().to(device_syn)
embedding_reconstruction_criterion = nn.L1Loss().to(device_syn)
rPPG_NP = NegPearsonLoss().to(device_syn)
rPPG_L1 = nn.L1Loss().to(device_syn)
rPPG_var = VarianceLoss().to(device_syn)
rPPG_feature_consistency_criterion = nn.L1Loss().to(device_rppg)
print("loss done")

epoch_rppg = 0
epoch_de = 0
epoch_embed = 0
# rPPG net checkpoint
try:
    latest_pkl_path = sorted(os.listdir(f'./{real_rppg_weight_folder_name}'))[-1]
    checkpoint = torch.load(f'./{real_rppg_weight_folder_name}/{latest_pkl_path}',  map_location=device_rppg)
    Physnet_rppg.load_state_dict(checkpoint['model_state_dict'])
    optimizer_real_rPPG.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_rppg = checkpoint['epoch']
    print(f'load the {epoch_rppg}th epoch from real_rppg')
except:
    print('no model found for real_rppg')
    epoch_rppg = 0
# Removal net checkpoint
try:
    latest_pkl_path = sorted(os.listdir(f'./{de_rppg_weight_folder_name}'))[-1]
    checkpoint = torch.load(f'./{de_rppg_weight_folder_name}/{latest_pkl_path}', map_location=device_syn)
    De_rPPG_encoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer_de_rPPG.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_de = checkpoint['epoch']
    print(f'load the {epoch_de}th epoch from de_rppg')
    epoch_de += 1
    print(f'the {epoch_de}th epoch from de_rppg')
except:
    print('no model found for de_rppg, use pretrained one')
    epoch_de = 0
# embed net checkpoint
try:
    latest_pkl_path = sorted(os.listdir(f'./{embed_rppg_weight_folder_name}'))[-1]
    checkpoint = torch.load(f'./{embed_rppg_weight_folder_name}/{latest_pkl_path}', map_location=device_syn)
    Rppg_embeder.load_state_dict(checkpoint['model_state_dict'])
    optimizer_embed_rPPG.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_embed = checkpoint['epoch']
    print(f'load the {epoch_embed}th epoch from embed_rppg')
    epoch_embed += 1
    print(f'the {epoch_embed}th epoch from embed_rppg')
except:
    print('no model found for embed_rppg, use pretrained one')
    epoch_embed = 0


while epoch_de <= 3501:

    print("epoch_rppg:", epoch_rppg)
    print("epoch_de:", epoch_de)
    print("epoch_embed:", epoch_embed)

    train_running_loss = 0
    train_running_removal_consistency_loss = 0
    train_running_embedding_consistency_loss = 0
    train_running_removal_reconstruction_loss = 0
    train_running_embedding_reconstruction_loss = 0
    train_running_source_rppg_NP_loss = 0
    train_running_source_consistency_loss = 0
    train_running_de_rppg_consistency_loss = 0
    train_running_de_rppg_P_loss = 0
    train_running_embed_rppg_loss = 0
    train_running_embed_target_rppg_loss = 0
    train_running_embed_rppg_consistency_loss = 0
    train_running_de_target_rppg_loss = 0
    train_running_source_feature_consistency_loss = 0
    train_running_removal_feature_consistency_loss = 0
    train_running_target_rppg_loss = 0
    train_running_removal_rppg_loss = 0

    #####################################
    for batch, sample_batched in enumerate(trainloader):
        Physnet_rppg.eval()
        De_rPPG_encoder.train()
        Rppg_embeder.train()

        for param in Physnet_rppg.parameters():
            param.requires_grad = False

        for param in De_rPPG_encoder.parameters():
            param.requires_grad = True

        for param in Rppg_embeder.parameters():
            param.requires_grad = True

        optimizer_de_rPPG.zero_grad()
        optimizer_embed_rPPG.zero_grad()

        De_rPPG_encoder.zero_grad()
        Rppg_embeder.zero_grad()

        target_rppg_label = get_fake_rppg(fake_label_set).repeat(1,1)
        rppg_label = sample_batched['label'].repeat(1,1)
        

        #input(real) -> removal -> embedding(source) -> removal
        with torch.no_grad():
            real_rppg, map1 = Physnet_rppg(sample_batched['face_frames'].to(device_rppg))
        no_rppg, _ = Physnet_rppg(sample_batched['no_face_frames'].to(device_rppg))
        de_rppg_face, de_rppg_att_map, de_rppg_signal_video = De_rPPG_encoder(sample_batched['face_frames'].to(device_syn), no_rppg.to(device_syn))
        de_rppg, map2 = Physnet_rppg(de_rppg_face.to(device_rppg))
        source_face, source_face_att_map, source_face_signal_video = Rppg_embeder(de_rppg_face, rppg_label.to(device_syn))
        source_rppg, map3 = Physnet_rppg(source_face.to(device_rppg))
        de_source_face, de_source_att_map, de_source_signal_video = De_rPPG_encoder(source_face,no_rppg.to(device_syn))
        de_source_rppg, map4 = Physnet_rppg(de_source_face.to(device_rppg))
        
        #removal consistency loss
        removal_consistency_loss = sigmoid_rampup(epoch_de, rppg_rampup)*consistency_weight*removal_consistency_criterion(de_rppg_face.to(device_syn),de_source_face.to(device_syn))
        #embedding reconstruction loss
        embedding_reconstruction_loss = reconstruction_weight*embedding_reconstruction_criterion(sample_batched['face_frames'].to(device_syn), source_face.to(device_syn))

        #feature consistency loss
        source_rPPG_feature_consistency_loss = 0.0
        for i in range(len(map1)):
            source_rPPG_feature_consistency_loss += feature_consistency_weight*rPPG_feature_consistency_criterion(map1[i].to(device_rppg),map3[i].to(device_rppg))
        feature_consistency_loss1 = source_rPPG_feature_consistency_loss 
        feature_consistency_loss1.backward(retain_graph=True)

        #rppg loss
        source_rppg_NP_loss = rPPG_NP(source_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn), rppg_label.to(device_syn))
        source_rppg_consistency_loss = 10.0*rPPG_L1(source_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn), real_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn))
        de_rppg_consistency_loss = 10.0*rPPG_L1(de_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn), de_source_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn))
        de_rppg_P_loss = rPPG_NP(de_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn), no_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn))
        de_source_rppg_P_loss = rPPG_NP(de_source_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn),no_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn))
        rppg_loss1 = source_rppg_NP_loss + de_rppg_P_loss + de_rppg_consistency_loss + de_source_rppg_P_loss + source_rppg_consistency_loss 

        #total loss
        loss1 = removal_consistency_loss + embedding_reconstruction_loss + rppg_weight*rppg_loss1
        
        loss1.backward()

        #input(static) -> embedding(target) -> removal -> embedding(target)
        single_frame = sample_batched['face_frames'][:,:,0] # [1, 3, 200, 200]
        static_face = single_frame.unsqueeze(2).repeat(1, 1, seq_length, 1, 1) # # [1, 3, 200, 200] -> # [1, 3, 100, 200, 200]
        single_no_face_frame = sample_batched['no_face_frames'][:,:,0] # [1, 3, 200, 200]
        static_no_face = single_no_face_frame.unsqueeze(2).repeat(1, 1, seq_length, 1, 1) # # [1, 3, 200, 200] -> # [1, 3, 100, 200, 200]
        with torch.no_grad():
            static_rppg, map5 = Physnet_rppg(static_face.to(device_rppg))
        static_no_rppg, _= Physnet_rppg(static_no_face.to(device_rppg))
        embed_rppg_face, embed_rppg_att_map, embed_rppg_signal_video = Rppg_embeder(static_face.to(device_syn), target_rppg_label.to(device_syn))
        embed_rppg, map6 = Physnet_rppg(embed_rppg_face.to(device_rppg))
        de_target_face, de_target_face_att_map, de_target_face_signal_video = De_rPPG_encoder(embed_rppg_face, static_no_rppg.to(device_syn))
        de_target_rppg, map7 = Physnet_rppg(de_target_face.to(device_rppg))
        embed_target_rppg_face, embed_target_att_map, embed_target_signal_video = Rppg_embeder(de_target_face, target_rppg_label.to(device_syn))
        embed_target_rppg, map8 = Physnet_rppg(embed_target_rppg_face.to(device_rppg))

        #embedding consistency loss
        embedding_consistency_loss = sigmoid_rampup(epoch_de, rppg_rampup)*consistency_weight*embedding_consistency_criterion(embed_rppg_face.to(device_syn), embed_target_rppg_face.to(device_syn))
        
        #removal reconstruction loss
        removal_reconstruction_loss = reconstruction_weight*removal_reconstruction_criterion(static_face.to(device_syn), de_target_face.to(device_syn))

        #feature consistency loss
        removal_rPPG_feature_consistency_loss = 0.0
        for i in range(len(map5)):
            removal_rPPG_feature_consistency_loss += feature_consistency_weight*rPPG_feature_consistency_criterion(map5[i].to(device_rppg),map7[i].to(device_rppg))
        feature_consistency_loss2 = removal_rPPG_feature_consistency_loss 
        feature_consistency_loss2.backward(retain_graph=True)
        
        #rppg loss
        embed_rppg_loss = rPPG_NP(embed_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn), target_rppg_label.to(device_syn))
        embed_target_rppg_loss = rPPG_NP(embed_target_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn), target_rppg_label.to(device_syn))
        embed_rppg_consistency_loss = 10.0*rPPG_L1(embed_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn), embed_target_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn))
        de_target_rppg_L1_loss = 10.0*rPPG_L1(de_target_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn), static_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn))
        de_target_rppg_var_loss = 100.0*rPPG_var(de_target_rppg.to(device_syn))
        rppg_loss2 = embed_rppg_loss + de_target_rppg_L1_loss + embed_target_rppg_loss + embed_rppg_consistency_loss + de_target_rppg_var_loss 

        #total loss
        loss2 = embedding_consistency_loss + removal_reconstruction_loss + rppg_weight*rppg_loss2 
        
        loss2.backward()

        #input(real) -> removal -> embed(target)
        no_rppg2, _ = Physnet_rppg(sample_batched['no_face_frames'].to(device_rppg))
        de_rppg_face2, de_rppg_att_map2, de_rppg_signal_video2 = De_rPPG_encoder(sample_batched['face_frames'].to(device_syn), no_rppg2.to(device_syn))
        removal_rppg, _ = Physnet_rppg(de_rppg_face2.to(device_rppg))
        embed_rppg_face2, embed_rppg_att_map2, embed_rppg_signal_video2 = Rppg_embeder(de_rppg_face2.to(device_syn), target_rppg_label.to(device_syn))
        target_rppg, _ = Physnet_rppg(embed_rppg_face2.to(device_rppg))

        #embedding loss
        embed_source_target_rppg_loss = rPPG_NP(target_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn), target_rppg_label.to(device_syn))

        #removal loss
        removal_source_static_rppg_loss = rPPG_NP(removal_rppg.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn), no_rppg2.squeeze(1).squeeze(-1).squeeze(-1).to(device_syn))

        #total loss
        loss3 = embed_source_target_rppg_loss + removal_source_static_rppg_loss
        loss3.backward()
        

        optimizer_de_rPPG.step()
        optimizer_embed_rPPG.step()
        
        train_running_removal_consistency_loss += removal_consistency_loss.item()
        train_running_embedding_reconstruction_loss += embedding_reconstruction_loss.item()
        train_running_source_rppg_NP_loss += source_rppg_NP_loss.item()
        train_running_source_consistency_loss +=source_rppg_consistency_loss.item()
        train_running_de_rppg_consistency_loss +=de_rppg_consistency_loss.item()
        train_running_de_rppg_P_loss = train_running_de_rppg_P_loss + de_rppg_P_loss.item() + de_source_rppg_P_loss.item()
        train_running_source_feature_consistency_loss += source_rPPG_feature_consistency_loss.item()
    
        train_running_loss += (loss1.item() + feature_consistency_loss1.item())

        train_running_embedding_consistency_loss += embedding_consistency_loss.item()
        train_running_removal_reconstruction_loss += removal_reconstruction_loss.item()
        train_running_embed_rppg_loss += embed_rppg_loss.item()
        train_running_embed_target_rppg_loss += embed_target_rppg_loss.item()
        train_running_embed_rppg_consistency_loss += embed_rppg_consistency_loss.item()
        train_running_de_target_rppg_loss = train_running_de_target_rppg_loss + de_target_rppg_L1_loss.item() + de_target_rppg_var_loss.item()
        train_running_removal_feature_consistency_loss += removal_rPPG_feature_consistency_loss.item()

        train_running_loss += (loss2.item() + feature_consistency_loss2.item())

        train_running_target_rppg_loss += embed_source_target_rppg_loss.item()
        train_running_removal_rppg_loss += removal_source_static_rppg_loss.item()
        train_running_loss += loss3.item()

        print(f"batch_train: {batch}/{len(trainloader)}")
        print('total_loss:', loss1.item()+loss2.item()+feature_consistency_loss1.item()+feature_consistency_loss2.item()+loss3.item())
        print('\tremoval_consistency_loss:',removal_consistency_loss.item())
        print('\tembedding_consistency_loss:',embedding_consistency_loss.item())
        print('\tremoval_reconstruction_loss:',removal_reconstruction_loss.item())
        print('\tembedding_reconstruction_loss:',embedding_reconstruction_loss.item())
        print('\tsource_rppg_NP_loss:',source_rppg_NP_loss.item())
        print('\tsource_consistency_loss:',source_rppg_consistency_loss.item())
        print('\tde_rppg_consistency_loss:',de_rppg_consistency_loss.item())
        print('\tde_rppg_P_loss:',de_rppg_P_loss.item() + de_source_rppg_P_loss.item())
        print('\tembed_rppg_loss:',embed_rppg_loss.item())
        print('\tembed_target_rppg_loss:',embed_target_rppg_loss.item())
        print('\tembed_rppg_consistency_loss:',embed_rppg_consistency_loss.item())
        print('\tde_target_rppg_loss:',de_target_rppg_L1_loss.item() + de_target_rppg_var_loss.item())
        print('\tsource_rppg_feature_consistency_loss:',source_rPPG_feature_consistency_loss.item())
        print('\tremoval_rppg_feature_consistency_loss:',removal_rPPG_feature_consistency_loss.item())
        print('\tembed_source_target_rppg_loss:',embed_source_target_rppg_loss.item())
        print('\tremoval_source_static_rppg_loss:',removal_source_static_rppg_loss.item())

    torch.save({
        'epoch': epoch_de,
        'model_state_dict': De_rPPG_encoder.state_dict(),
        'optimizer_state_dict': optimizer_de_rPPG.state_dict()
    }, f'./{de_rppg_weight_folder_name}/{"{:0>4d}".format(epoch_de)}_{"{:0>4d}".format(int(len(trainloader) / train_batch_size))}.pkl')
    torch.save({
        'epoch': epoch_embed,
        'model_state_dict': Rppg_embeder.state_dict(),
        'optimizer_state_dict': optimizer_embed_rPPG.state_dict()
    }, f'./{embed_rppg_weight_folder_name}/{"{:0>4d}".format(epoch_embed)}_{"{:0>4d}".format(int(len(trainloader) / train_batch_size))}.pkl')
    with open(f'{network_name}_training_loss_de_rppg.txt', 'a') as f:
        f.write(f'epoch_rppg:{epoch_rppg}, epoch_de:{epoch_de}, epoch_embed:{epoch_embed}:\n')
        f.write(f'total_loss:{"{0:.4f}".format(train_running_loss / (len(trainloader)))}\n')
        f.write(f'\tremoval_consistency_loss:{"{0:.4f}".format(train_running_removal_consistency_loss / (len(trainloader)))}\n')
        f.write(f'\tremoval_reconstruction_loss:{"{0:.4f}".format(train_running_removal_reconstruction_loss / (len(trainloader)))}\n')
        f.write(f'\tembedding_consistency_loss:{"{0:.4f}".format(train_running_embedding_consistency_loss / (len(trainloader)))}\n')
        f.write(f'\tembedding_reconstruction_loss:{"{0:.4f}".format(train_running_embedding_reconstruction_loss / (len(trainloader)))}\n')
        f.write(f'\tsource_rppg_NP_loss:{"{0:.4f}".format(train_running_source_rppg_NP_loss / (len(trainloader)))}\n')
        f.write(f'\tsource_consistency_loss:{"{0:.4f}".format(train_running_source_consistency_loss / (len(trainloader)))}\n')
        f.write(f'\tde_rppg_consistency_loss:{"{0:.4f}".format(train_running_de_rppg_consistency_loss / (len(trainloader)))}\n')
        f.write(f'\tde_rppg_P_loss:{"{0:.4f}".format(train_running_de_rppg_P_loss / (len(trainloader)))}\n')
        f.write(f'\tembed_rppg_loss:{"{0:.4f}".format(train_running_embed_rppg_loss / (len(trainloader)))}\n')
        f.write(f'\tembed_target_rppg_loss:{"{0:.4f}".format(train_running_embed_target_rppg_loss / (len(trainloader)))}\n')
        f.write(f'\tembed_rppg_consistency_loss:{"{0:.4f}".format(train_running_embed_rppg_consistency_loss / (len(trainloader)))}\n')
        f.write(f'\tde_target_rppg_loss:{"{0:.4f}".format(train_running_de_target_rppg_loss / (len(trainloader)))}\n')
        f.write(f'\tsource_rppg_feature_consistency_loss:{"{0:.4f}".format(train_running_source_feature_consistency_loss / (len(trainloader)))}\n')
        f.write(f'\tremoval_rppg_feature_consistency_loss:{"{0:.4f}".format(train_running_removal_feature_consistency_loss / (len(trainloader)))}\n')
        f.write(f'\tembed_source_target_rppg_loss:{"{0:.4f}".format(train_running_target_rppg_loss / (len(trainloader)))}\n')
        f.write(f'\tremoval_source_target_rppg_loss:{"{0:.4f}".format(train_running_removal_rppg_loss / (len(trainloader)))}\n')

    epoch_embed += 1
    epoch_de += 1
