"""
@Time    : 22/06/2022 12:33
@Author  : Colm Keyes, Luc Dael
@Email   : keyesco@tcd.ie, luc.dael@wur.nl
@File    : UISB.py
@Paper   : https://ieeexplore.ieee.org/document/8462533
@Credits : Asako Kanezak, AIST, Tokyo, Japan.
@License : MIT
"""

##############
## Unsupervised Image Segmentation by Backpropegation
##############



import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import seaborn as sns
import numpy as np
from skimage import segmentation
import torch.nn.init
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'



##############
## Inits
##############

##CUDA init
#use_cuda = torch.cuda.is_available()

input_file = 'C:/Users/Colm The Creator/Desktop/pytorch-unsupervised-segmentation-478e67e46948216de82ea0bc6ba242ad6a79d442/orthoHRmasked.tif'
minLabels = 25
nconv_range = 3   # min is 2
compactness_array = [2,3,4,6,9]#8,16,32,64] #0.5,0.6,0.75,0.85] # array of compactness to be set

# File Inits
image_directory = 'Results/Image_output/best_params-classes ' + str(minLabels)
class_array_directory = 'Results/class_Numpy_output/best_params-classes ' + str(minLabels)

for directory in [image_directory,class_array_directory]:
    if not os.path.exists(directory):
        os.makedir(directory)


# ArgParser - useful for initing from terminal, this is also where default values are set for the network

def define_args(input_image):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
    parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                        help='number of channels')
    parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                        help='number of maximum iterations')
    parser.add_argument('--minLabels', metavar='minL', default=25, type=int,
                        help='minimum number of labels')
    parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                        help='learning rate')
    parser.add_argument('--nConv', metavar='M', default=2, type=int,
                        help='number of convolutional layers')
    parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
                        help='number of superpixels')
    parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                        help='visualization flag')
    parser.add_argument('--compactness', metavar='C', default=100, type=float,
                        help='compactness of superpixels')
    parser.add_argument('--input', metavar='FILENAME',
                        help='input image file name',default=input_image, required=False) # True)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    args = parser.parse_args()
    return args


# UISB model
class UISB(nn.Module):
    def __init__(self,input_dim):
        super(UISB, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

def train(args):
    im = cv2.imread(args.input)
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    #if use_cuda:
    #    data = data.cuda()
    data = Variable(data)

    # slic
    labels = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

    # train
    model = UISB( data.size(1) )
    #if use_cuda:
    #    model.cuda()
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    palet = sns.color_palette(n_colors=100)
    palet_df = pd.DataFrame(palet)*255
    label_colours = palet_df.to_numpy()

    for batch_idx in range(args.maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        if args.visualize:
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )

            #cv2.imshow( "output", im_target_rgb ) ## uncomment to visualise

        # superpixel refinement
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
        target = torch.from_numpy( im_target )
        #if use_cuda:
        #    target = target.cuda()
        target = Variable( target )
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())

        ########
        ## How a conclusion is reached
        #######
        if nLabels <= args.minLabels:
           print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
           break

    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )

    cv2.imwrite( image_directory+'/output-image minlabels '+str(args.minLabels)+
                 ' compactness '+str(args.compactness)+' LR '+str(args.lr)+
                ' nConv '+str(args.nConv)+' nIter ' +str(batch_idx)+
                ' time-'+str(time.time())+'.tif', im_target_rgb )
    np.save(class_array_directory+'/tensor-output minlabels '+str(args.minLabels)+
            ' compactness '+str(args.compactness)+' LR '+str(args.lr)+
            ' nIter ' +str(batch_idx)+
            ' time-'+str(time.time()),im_target.reshape((im[:,0,0].__len__(), im[0,:,0].__len__(),1)))


if __name__ == '__main__':
    args = define_args(input_file)
    args.minLabels = minLabels
    #args
    for i in range(3, nconv_range+1): ## Change this range value
        for n in np.array(compactness_array):
            args.nConv = i
            args.compactness = n
            print('nConv = '+str(args.nConv))
            print('Compactness = '+str(args.compactness))
            train(args)