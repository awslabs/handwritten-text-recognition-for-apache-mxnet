# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import random
import os
import matplotlib.pyplot as plt
import argparse

import mxnet as mx
import numpy as np
from skimage import transform as skimage_tf
from skimage import exposure

from mxnet import nd, autograd, gluon
from mxboard import SummaryWriter
from mxnet.gluon.model_zoo.vision import resnet34_v1
np.seterr(all='raise')

import multiprocessing
mx.random.seed(1)

from .utils.iam_dataset import IAMDataset, resize_image
from .utils.draw_text_on_image import draw_text_on_image

print_every_n = 1
send_image_every_n = 20

# Best results:
# python handwriting_line_recognition.py --epochs 251 -n handwriting_line.params -g 0 -l 0.0001 -x 0.1 -y 0.1 -j 0.15 -k 0.15 -p 0.75 -o 2 -a 128

alphabet_encoding = r' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
alphabet_dict = {alphabet_encoding[i]:i for i in range(len(alphabet_encoding))}

class EncoderLayer(gluon.HybridBlock):
    '''
    The encoder layer takes the image features from a CNN. The image features are transposed so that the LSTM 
    slices of the image features can be sequentially fed into the LSTM from left to right (and back via the
    bidirectional LSTM). 
    '''
    def __init__(self, hidden_states=200, rnn_layers=1, max_seq_len=100, **kwargs):
        self.max_seq_len = max_seq_len
        super(EncoderLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.lstm = mx.gluon.rnn.LSTM(hidden_states, rnn_layers, bidirectional=True)
            
    def hybrid_forward(self, F, x):
        x = x.transpose((0, 3, 1, 2))
        x = x.flatten()
        x = x.split(num_outputs=self.max_seq_len, axis=1) # (SEQ_LEN, N, CHANNELS)
        x = F.concat(*[elem.expand_dims(axis=0) for elem in x], dim=0)
        x = self.lstm(x)
        x = x.transpose((1, 0, 2)) #(N, SEQ_LEN, HIDDEN_UNITS)
        return x

class Network(gluon.HybridBlock):
    '''
    The CNN-biLSTM to recognise handwriting text given an image of handwriten text.
    Parameters
    ----------
    num_downsamples: int, default 2
        The number of times to downsample the image features. Each time the features are downsampled, a new LSTM
        is created. 
    resnet_layer_id: int, default 4
        The layer ID to obtain features from the resnet34
    lstm_hidden_states: int, default 200
        The number of hidden states used in the LSTMs
    lstm_layers: int, default 1
        The number of layers of LSTMs to use
    '''
    FEATURE_EXTRACTOR_FILTER = 64
    def __init__(self, num_downsamples=2, resnet_layer_id=4, rnn_hidden_states=200, rnn_layers=1, max_seq_len=100, ctx=mx.gpu(0), **kwargs):
        super(Network, self).__init__(**kwargs)
        self.p_dropout = 0.5
        self.num_downsamples = num_downsamples
        self.max_seq_len = max_seq_len
        self.ctx = ctx
        with self.name_scope():
            self.body = self.get_body(resnet_layer_id=resnet_layer_id)

            self.encoders = gluon.nn.HybridSequential()
            with self.encoders.name_scope():
                for i in range(self.num_downsamples):
                    encoder = self.get_encoder(rnn_hidden_states=rnn_hidden_states, rnn_layers=rnn_layers, max_seq_len=max_seq_len)
                    self.encoders.add(encoder)
            self.decoder = self.get_decoder()
            self.downsampler = self.get_down_sampler(self.FEATURE_EXTRACTOR_FILTER)

    def get_down_sampler(self, num_filters):
        '''
        Creates a two-stacked Conv-BatchNorm-Relu and then a pooling layer to
        downsample the image features by half.
        
        Parameters
        ----------
        num_filters: int
            To select the number of filters in used the downsampling convolutional layer.
        Returns
        -------
        network: gluon.nn.HybridSequential
            The downsampler network that decreases the width and height of the image features by half.
        
        '''
        out = gluon.nn.HybridSequential()
        with out.name_scope():
            for _ in range(2):
                out.add(gluon.nn.Conv2D(num_filters, 3, strides=1, padding=1))
                out.add(gluon.nn.BatchNorm(in_channels=num_filters))
                out.add(gluon.nn.Activation('relu'))
            out.add(gluon.nn.MaxPool2D(2))
            out.collect_params().initialize(mx.init.Normal(), ctx=self.ctx)
        out.hybridize()
        return out

    def get_body(self, resnet_layer_id):
        '''
        Create the feature extraction network based on resnet34.
        The first layer of the res-net is converted into grayscale by averaging the weights of the 3 channels
        of the original resnet.
        
        Parameters
        ----------
        resnet_layer_id: int
            The resnet_layer_id specifies which layer to take from 
            the bottom of the network.
        Returns
        -------
        network: gluon.nn.HybridSequential
            The body network for feature extraction based on resnet
        '''
        
        pretrained = resnet34_v1(pretrained=True, ctx=self.ctx)
        pretrained_2 = resnet34_v1(pretrained=True, ctx=mx.cpu(0))
        first_weights = pretrained_2.features[0].weight.data().mean(axis=1).expand_dims(axis=1)
        # First weights could be replaced with individual channels.
        
        body = gluon.nn.HybridSequential()
        with body.name_scope():
            first_layer = gluon.nn.Conv2D(channels=64, kernel_size=(7, 7), padding=(3, 3), strides=(2, 2), in_channels=1, use_bias=False)
            first_layer.initialize(mx.init.Xavier(), ctx=self.ctx)
            first_layer.weight.set_data(first_weights)
            body.add(first_layer)
            body.add(*pretrained.features[1:-resnet_layer_id])
        return body

    def get_encoder(self, rnn_hidden_states, rnn_layers, max_seq_len):
        '''
        Creates an LSTM to learn the sequential component of the image features.
        
        Parameters
        ----------
        
        rnn_hidden_states: int
            The number of hidden states in the RNN
        
        rnn_layers: int
            The number of layers to stack the RNN
        Returns
        -------
        
        network: gluon.nn.Sequential
            The encoder network to learn the sequential information of the image features
        '''

        encoder = gluon.nn.HybridSequential()
        with encoder.name_scope():
            encoder.add(EncoderLayer(hidden_states=rnn_hidden_states, rnn_layers=rnn_layers, max_seq_len=max_seq_len))
            encoder.add(gluon.nn.Dropout(self.p_dropout))
        encoder.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        return encoder
    
    def get_decoder(self):
        '''
        Creates a network to convert the output of the encoder into characters.
        '''

        alphabet_size = len(alphabet_encoding) + 1
        decoder = mx.gluon.nn.Dense(units=alphabet_size, flatten=False)
        decoder.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        return decoder

    def hybrid_forward(self, F, x):
        features = self.body(x)
        hidden_states = []
        hs = self.encoders[0](features)
        hidden_states.append(hs)
        for i, _ in enumerate(range(self.num_downsamples - 1)):
            features = self.downsampler(features)
            hs = self.encoders[i+1](features)
            hidden_states.append(hs)
        hs = F.concat(*hidden_states, dim=2)
        output = self.decoder(hs)
        return output

def handwriting_recognition_transform(image, line_image_size):
    '''
    Resize and normalise the image to be fed into the network.
    '''
    image, _ = resize_image(image, line_image_size)
    image = mx.nd.array(image)/255.
    image = (image - 0.942532484060557) / 0.15926149044640417
    image = image.expand_dims(0).expand_dims(0)
    return image

def transform(image, label):
    '''
    This function resizes the input image and converts so that it could be fed into the network.
    Furthermore, the label (text) is one-hot encoded.
    '''
    image = np.expand_dims(image, axis=0).astype(np.float32)
    if image[0, 0, 0] > 1:
        image = image/255.
    image = (image - 0.942532484060557) / 0.15926149044640417
    label_encoded = np.zeros(max_seq_len, dtype=np.float32)-1
    i = 0
    for word in label:
        word = word.replace("&quot", r'"')
        word = word.replace("&amp", r'&')
        word = word.replace('";', '\"')
        for letter in word:
            label_encoded[i] = alphabet_dict[letter]
            i += 1
    return image, label_encoded

def augment_transform(image, label):
    '''
    This function randomly:
        - translates the input image by +-width_range and +-height_range (percentage).
        - scales the image by y_scaling and x_scaling (percentage)
        - shears the image by shearing_factor (radians)
    '''

    ty = random.uniform(-random_y_translation, random_y_translation)
    tx = random.uniform(-random_x_translation, random_x_translation)

    sx = random.uniform(1. - random_y_scaling, 1. + random_y_scaling)
    sy = random.uniform(1. - random_x_scaling, 1. + random_x_scaling)

    s = random.uniform(-random_shearing, random_shearing)
    gamma = random.uniform(0.001, random_gamma)
    image = exposure.adjust_gamma(image, gamma)

    st = skimage_tf.AffineTransform(scale=(sx, sy),
                                    shear=s,
                                    translation=(tx*image.shape[1], ty*image.shape[0]))
    augmented_image = skimage_tf.warp(image, st, cval=1.0)
    return transform(augmented_image*255., label)


def decode(prediction):
    '''
    Returns the string given one-hot encoded vectors.
    '''

    results = []
    for word in prediction:
        result = []
        for i, index in enumerate(word):
            if i < len(word) - 1 and word[i] == word[i+1] and word[-1] != -1: #Hack to decode label as well
                continue
            if index == len(alphabet_dict) or index == -1:
                continue
            else:
                result.append(alphabet_encoding[int(index)])
        results.append(result)
    words = [''.join(word) for word in results]
    return words

def run_epoch(e, network, dataloader, trainer, log_dir, print_name, is_train):
    '''
    Run one epoch to train or test the CNN-biLSTM network
    
    Parameters
    ----------
        
    e: int
        The epoch number
    network: nn.Gluon.HybridSequential
        The CNN-biLSTM network
    dataloader: gluon.data.DataLoader
        The train or testing dataloader that is wrapped around the iam_dataset
    
    log_dir: Str
        The directory to store the log files for mxboard
    print_name: Str
        Name to print for associating with the data. usually this will be "train" and "test"
    
    is_train: bool
        Boolean to indicate whether or not the network should be updated. is_train should only be set to true for the training data
    Returns
    -------
    
    epoch_loss: float
        The loss of the current epoch
    '''

    total_loss = [nd.zeros(1, ctx_) for ctx_ in ctx]
    for i, (x_, y_) in enumerate(dataloader):
        X = gluon.utils.split_and_load(x_, ctx)
        Y = gluon.utils.split_and_load(y_, ctx)
        with autograd.record(train_mode=is_train):
            output = [network(x) for x in X]
            loss_ctc = [ctc_loss(o, y) for o, y in zip(output, Y)]

        if is_train:
            [l.backward() for l in loss_ctc]
            trainer.step(x_.shape[0])

        if i == 0 and e % send_image_every_n == 0 and e > 0:
            predictions = output[0][:4].softmax().topk(axis=2).asnumpy()
            decoded_text = decode(predictions)
            image = X[0][:4].asnumpy()
            image = image * 0.15926149044640417 + 0.942532484060557            
            output_image = draw_text_on_image(image, decoded_text)
            print("{} first decoded text = {}".format(print_name, decoded_text[0]))
            with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
                sw.add_image('bb_{}_image'.format(print_name), output_image, global_step=e)

        for i, l in enumerate(loss_ctc):
            total_loss[i] += l.mean()

    epoch_loss = float(sum([tl.asscalar() for tl in total_loss]))/(len(dataloader)*len(ctx))

    with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
        sw.add_scalar('loss', {print_name: epoch_loss}, global_step=e)

    return epoch_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", default="0",
                        help="IDs of the GPU to use, -1 for CPU")

    parser.add_argument("-t", "--line_or_word", default="line",
                        help="to choose the handwriting to train on words or lines")

    parser.add_argument("-u", "--num_downsamples", default=2,
                        help="Number of downsamples for the res net")
    parser.add_argument("-q", "--resnet_layer_id", default=4,
                        help="layer ID to obtain features from the resnet34")
    parser.add_argument("-a", "--rnn_hidden_states", default=200,
                        help="Number of hidden states for the RNN encoder")
    parser.add_argument("-o", "--rnn_layers", default=1,
                        help="Number of layers for the RNN")

    parser.add_argument("-e", "--epochs", default=121,
                        help="Number of epochs to run")
    parser.add_argument("-l", "--learning_rate", default=0.0001,
                        help="Learning rate for training")
    parser.add_argument("-w", "--lr_scale", default=1,
                        help="Amount the divide the learning rate")
    parser.add_argument("-r", "--lr_period", default=30,
                        help="Divides the learning rate after period")

    parser.add_argument("-s", "--batch_size", default=64,
                        help="Batch size")

    parser.add_argument("-x", "--random_x_translation", default=0.03,
                        help="Randomly translation the image in the x direction (+ or -)")
    parser.add_argument("-y", "--random_y_translation", default=0.03,
                        help="Randomly translation the image in the y direction (+ or -)")
    parser.add_argument("-j", "--random_x_scaling", default=0.10,
                        help="Randomly scale the image in the x direction")
    parser.add_argument("-k", "--random_y_scaling", default=0.10,
                        help="Randomly scale the image in the y direction")
    parser.add_argument("-p", "--random_shearing", default=0.5,
                        help="Randomly shear the image in radians (+ or -)")
    parser.add_argument("-ga", "--random_gamma", default=1,
                        help="Randomly update gamma of image (+ or -)")

    parser.add_argument("-d", "--log_dir", default="./logs",
                        help="Directory to store the log files")
    parser.add_argument("-c", "--checkpoint_dir", default="model_checkpoint",
                        help="Directory to store the checkpoints")
    parser.add_argument("-n", "--checkpoint_name", default="handwriting_line.params",
                        help="Name to store the checkpoints")
    parser.add_argument("-m", "--load_model", default=None,
                         help="Name of model to load")
    parser.add_argument("-sl", "--max-seq-len", default=None,
                         help="Maximum sequence length")
    args = parser.parse_args()

    print(args)
    
    gpu_ids = [int(elem) for elem in args.gpu_id.split(",")]
    
    if gpu_ids == [-1]:
        ctx=[mx.cpu()]
    else:
        ctx=[mx.gpu(i) for i in gpu_ids]

    line_or_word = args.line_or_word
    assert line_or_word in ["line", "word"], "{} is not a value option in [\"line\", \"word\"]"
        
    num_downsamples = int(args.num_downsamples)
    resnet_layer_id = int(args.resnet_layer_id)
    rnn_hidden_states = int(args.rnn_hidden_states)
    rnn_layers = int(args.rnn_layers)
    
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    lr_scale = float(args.lr_scale)
    lr_period = float(args.lr_period)
    batch_size = int(args.batch_size)
    
    random_y_translation, random_x_translation = float(args.random_x_translation), float(args.random_y_translation)
    random_y_scaling, random_x_scaling = float(args.random_y_scaling), float(args.random_x_scaling)
    random_shearing = float(args.random_shearing)
    random_gamma = float(args.random_gamma)
    
    log_dir = args.log_dir
    checkpoint_dir, checkpoint_name = args.checkpoint_dir, args.checkpoint_name
    load_model = args.load_model
    max_seq_len = args.max_seq_len
    
    if max_seq_len is not None:
        max_seq_len = int(max_seq_len)
    elif line_or_word == "line":
        max_seq_len = 100
    else:
        max_seq_len = 32
    
    net = Network(num_downsamples=num_downsamples, resnet_layer_id=resnet_layer_id , rnn_hidden_states=rnn_hidden_states, rnn_layers=rnn_layers,
                  max_seq_len=max_seq_len, ctx=ctx)

    if load_model is not None and os.path.isfile(os.path.join(checkpoint_dir,load_model)):
        net.load_parameters(os.path.join(checkpoint_dir,load_model))
        
    train_ds = IAMDataset(line_or_word, output_data="text", train=True)
    print("Number of training samples: {}".format(len(train_ds)))
    
    test_ds = IAMDataset(line_or_word, output_data="text", train=False)
    print("Number of testing samples: {}".format(len(test_ds)))
    
    train_data = gluon.data.DataLoader(train_ds.transform(augment_transform), batch_size, shuffle=True, last_batch="rollover", num_workers=4*len(ctx))
    test_data = gluon.data.DataLoader(test_ds.transform(transform), batch_size, shuffle=True, last_batch="discard", num_workers=4*len(ctx))
    
    schedule = mx.lr_scheduler.FactorScheduler(step=lr_period*len(train_data), factor=lr_scale)
    schedule.base_lr = learning_rate

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, "lr_scheduler": schedule, 'clip_gradient': 2})
    
    ctc_loss = gluon.loss.CTCLoss()
    
    best_test_loss = 10e10
    for e in range(epochs):
        train_loss = run_epoch(e, net, train_data, trainer, log_dir, print_name="train", is_train=True)
        test_loss = run_epoch(e, net, test_data, trainer, log_dir, print_name="test", is_train=False)    
        if test_loss < best_test_loss:
            print("Saving network, previous best test loss {:.6f}, current test loss {:.6f}".format(best_test_loss, test_loss))
            net.save_parameters(os.path.join(checkpoint_dir, checkpoint_name))
            best_test_loss = test_loss

        if e % print_every_n == 0 and e > 0:
            print("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}".format(e, train_loss, test_loss))
