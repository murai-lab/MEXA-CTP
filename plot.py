import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from glob import glob

import sys
from utils.utils import save_json, read_json

parser = argparse.ArgumentParser(
        prog='Plot results',
        description='Plot training and test curve',
    )
parser.add_argument('--path', default='./results/', help='model path')
parser.add_argument('--phase', default='II', help='model path')


def getData(file, ee):
    dataframe = pd.read_csv(file)
    # print(dataframe['train_loss'][:ee])
    return np.array(dataframe['train_loss'][:ee]), np.array(dataframe['train_acc'][:ee]), np.array(dataframe['test_loss'][:ee]), np.array(dataframe['test_acc'][:ee]), np.array(dataframe['test_f1'][:ee]), np.array(dataframe['test_roc'][:ee])

def getLabel(file):
    return file.split('/')[-2].split('_id')[0]
  


def plot_metric(path, ee=100, visualize=False, save_img=True):
    tl, ta, vl, va, f1, roc = getData(path, ee)
    epoch = [i for i in range(ee)]
    label = getLabel(path)
    pos_f1 = np.argmax(f1)
    pos_roc = np.argmax(roc)

    print('f1', np.argmax(f1), np.max(f1))
    print('roc', np.argmax(roc), np.max(roc))

    plt.subplot(2, 2, 1)
    plt.title('Training Loss')
    plt.plot(epoch, tl, label=label, linewidth=2.0)
    plt.plot(np.argmin(tl), min(tl), 'bo')
    plt.axvline(pos_f1, color='red', label='f1')
    plt.axvline(pos_roc, color='green', label='roc')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title(f'Training Accuracy')

    plt.plot(epoch, ta, label=label, linewidth=2.0)
    plt.plot(np.argmax(ta), max(ta), 'bo')
    plt.axvline(pos_f1, color='red', label='f1')
    plt.axvline(pos_roc, color='green', label='roc')
        #plt.legend()

    plt.subplot(2, 2, 3)
    plt.title('Test Loss')

    plt.plot(epoch, vl, label=label, linewidth=2.0)
    plt.plot(np.argmin(vl), min(vl), 'bo')
    plt.axvline(pos_f1, color='red', label='f1')
    plt.axvline(pos_roc, color='green', label='roc')
        #plt.legend()

    plt.subplot(2, 2, 4)
    plt.title(f'Test Accuracy')
 
    plt.plot(epoch, va, label=label, linewidth=2.0)
    plt.plot(np.argmax(va), max(va), 'bo')
    plt.axvline(pos_f1, color='red', label='f1')
    plt.axvline(pos_roc, color='green', label='roc')

    data_dict = {
        'acc': f'{max(va)}@{np.argmax(va)+1}',
        'f1': f'{np.max(f1)}@{np.argmax(f1)+1}',
        'roc': f'{np.max(roc)}@{np.argmax(roc)+1}'
    }


    plt.tight_layout()

    if save_img:
        plt.savefig(f'{path.replace("history.csv", "curve.jpg")}')
        save_json(data_dict, f'{path.replace("history.csv", "summary.json")}')

    if visualize:
        plt.show()

def plot_all():
    args = parser.parse_args()
    paths = glob(f'{args.path}/train_phase_{args.phase}/*/history.csv')
    for path in paths:
        print(path.split('/')[-2])
        if len(glob(path.replace("history.csv", "summary.json"))) > 0:
            read_json(path.replace("history.csv", "summary.json"))
        else:
            plot_metric(path, ee=100, visualize=False, save_img=True)



if __name__ == '__main__':
    plot_all()
