import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--syn_seq_length', type=int, default=100, help='the nunber of frames during synthesis training, default = 100')
    parser.add_argument('--device_syn', type=str, default="cuda:1", help='the device which removal and embedding network locate, default is cuda:0')
    parser.add_argument('--device_rppg', type=str, default="cuda:0", help='the device which rPPG network locates, default is cuda:1')
    args = parser.parse_args()
    return args

def get_result_folder_name(args):
    result_folder_name = "baseline"
    print(f"result_folder_name: {result_folder_name}")
    return result_folder_name

def get_aug_result_folder_name(args):
    result_folder_name = get_result_folder_name(args)
    aug_result_folder_name = f"{result_folder_name}_T{args.aug_seq_length}"
    print(f"aug_result_folder_name: {aug_result_folder_name}")
    return aug_result_folder_name

def get_rppg_result_folder_name(args):
    if args.rppg_att_type == 'MyCBAM_v3':
        network_name = "rPPG_with_attention" # CHANGE
    elif args.rppg_att_type == 'without':
        network_name = "rPPG_without_attention" # CHANGE
    else:
        raise ValueError("att_type is wrong")
    rppg_result_folder_name = f"(rppg_{args.rppg_train_data_type}) {network_name}"
    print(f"rppg_result_folder_name: {rppg_result_folder_name}")
    return rppg_result_folder_name
