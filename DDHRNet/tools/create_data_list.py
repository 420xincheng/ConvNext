import os
import numpy as np
from tqdm import tqdm
import argparse


def split_sar_optic(root):
    save_image = os.path.join(root, 'label')
    image_list = os.listdir(save_image)

    np.random.shuffle(image_list)
    # train : valid = 1 : 1
    train_list = image_list[::2]
    valid_list = [x for x in image_list if x not in train_list]

    with open(os.path.join(root, 'trainM_list.txt'), 'w', encoding='utf-8') as f:
        for each in tqdm(train_list):
            f.write(root + '/GF2/' + os.path.splitext(each)[0] + '.jpg' + ' '
                    + root + '/GF3/' + os.path.splitext(each)[0] + '.jpg' + ' '
                    + root + '/label/' + os.path.splitext(each)[0] + '.png' + '\n')

    with open(os.path.join(root, 'valM_list.txt'), 'w', encoding='utf-8') as f:
        for each in tqdm(valid_list):
            f.write(root + '/GF2/' + os.path.splitext(each)[0] + '.jpg' + ' '
                    + root + '/GF3/' + os.path.splitext(each)[0] + '.jpg' + ' '
                    + root + '/label/' + os.path.splitext(each)[0] + '.png' + '\n')

def split_whu_data(root):
    # root = '/workspace/www/dataDir/WHU'
    train_list = os.listdir(os.path.join(root, 'train/lbl'))
    val_list = os.listdir(os.path.join(root, 'val/lbl'))
    np.random.shuffle(train_list)
    np.random.shuffle(val_list)

    with open(os.path.join(root, 'train_WHU_list.txt'), 'w', encoding='utf-8') as f:
        for each in tqdm(train_list):
            filename = os.path.splitext(each)[0]
            f.write(root + '/train/opt/' + filename + '.jif' + ' '
                    + root + '/train/sar/' + filename + '.jif' + ' '
                    + root + '/train/lbl/' + filename + '.jif' + '\n')
    with open(os.path.join(root, 'val_WHU_list.txt'), 'w', encoding='utf-8') as f:
        for each in tqdm(val_list):
            filename = os.path.splitext(each)[0]
            f.write(root + '/train/opt/' + filename + '.jif' + ' '
                    + root + '/train/sar/' + filename + '.jif' + ' '
                    + root + '/train/lbl/' + filename + '.jif' + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PaddleSeg create data list')
    parser.add_argument(
        '--dataset',
        help='choose which dataset to use (WHU, Korea-cloud, ...)',
        default='WHU',
        type=str)
    args = parser.parse_args()
    if args.dataset == 'WHU':
        root_whu = '/workspace/www/dataDir/WHU'
        split_whu_data(root=root_whu)
    elif args.dataset == 'Korea-cloud':
        root = '/workspace/MaShibin/DATA/korea/cloud'
        split_sar_optic(root)