import time
import random
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import string

import mxnet as mx
import numpy as np
from skimage import transform as skimage_tf

from mxnet import nd, autograd, gluon
from mxnet.image import resize_short
from mxboard import SummaryWriter
from mxnet.gluon.model_zoo.vision import resnet34_v1
np.seterr(all='raise')

import multiprocessing
mx.random.seed(1)

from utils.iam_dataset import IAMDataset

max_seq_len = 100
print_every_n = 1
save_every_n = 50
send_image_every_n = 5

from utils.iam_dataset import IAMDataset
from utils.draw_text_on_image import draw_text_on_image

alphabet_encoding = string.ascii_letters+string.digits+string.punctuation+' '
alphabet_dict = {alphabet_encoding[i]:i for i in range(len(alphabet_encoding))}

class EncoderLayer(gluon.Block):
    def __init__(self, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        with self.name_scope():
            hidden_states = 200
            lstm_layers = 2
            self.lstm = mx.gluon.rnn.LSTM(hidden_states, lstm_layers, bidirectional=True)
            
    def forward(self, x):
        x = x.transpose((0,3,1,2))
        x = x.flatten()
        x = x.split(num_outputs=max_seq_len, axis = 1) # (SEQ_LEN, N, CHANNELS)
        x = nd.concat(*[elem.expand_dims(axis=0) for elem in x], dim=0)
        x = self.lstm(x)
        x = x.transpose((1, 0, 2)) # (N, SEQ_LEN, HIDDEN_UNITS)
        return x

class Network(gluon.Block):
    def __init__(self, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.p_dropout = 0.5
        self.net = gluon.nn.Sequential()
        with self.name_scope():
            self.net.add(self.get_body())
            self.net.add(self.get_encoder())
            self.net.add(self.get_decoder())
        #self.net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        
    def get_body(self):
        '''
        Create the feature extraction network of the SSD based on resnet34.
        The first layer of the res-net is converted into grayscale by averaging the weights of the 3 channels
        of the original resnet.

        Returns
        -------
        network: gluon.nn.HybridSequential
            The body network for feature extraction based on resnet
        
        '''
        pretrained = resnet34_v1(pretrained=True, ctx=ctx)
        pretrained_2 = resnet34_v1(pretrained=True, ctx=mx.cpu(0))
        first_weights = pretrained_2.features[0].weight.data().mean(axis=1).expand_dims(axis=1)
        # First weights could be replaced with individual channels.
        
        body = gluon.nn.HybridSequential()
        with body.name_scope():
            first_layer = gluon.nn.Conv2D(channels=64, kernel_size=(7, 7), padding=(3, 3), strides=(2, 2), in_channels=1, use_bias=False)
            first_layer.initialize(mx.init.Normal(), ctx=ctx)
            first_layer.weight.set_data(first_weights)
            body.add(first_layer)
            body.add(*pretrained.features[1:-3])
        return body

    def get_encoder(self):
        encoder = gluon.nn.Sequential()
        encoder.add(EncoderLayer())
        encoder.add(gluon.nn.Dropout(self.p_dropout))
        encoder.collect_params().initialize(mx.init.Normal(), ctx=ctx)
        return encoder
    
    def get_decoder(self):
        alphabet_size = len(string.ascii_letters+string.digits+string.punctuation+' ') + 1
        decoder = mx.gluon.nn.Dense(units=alphabet_size, flatten=False)
        decoder.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        return decoder

    def forward(self, x):
        return self.net(x)

def augment_transform(image, label):
    ty = random.uniform(-random_y_translation, random_y_translation)
    tx = random.uniform(-random_x_translation, random_x_translation)

    st = skimage_tf.SimilarityTransform(translation=(tx*image.shape[1], ty*image.shape[0]))
    augmented_image = skimage_tf.warp(image, st, cval=1.0)
    return transform(augmented_image*255., label)

def transform(image, label):
    image = np.expand_dims(image, axis=0).astype(np.float32)/255.
    # image = resize_short(nd.array(image), 50)
    
    label_encoded = np.zeros(max_seq_len, dtype=np.float32)-1
    i = 0
    for word in label:
        # if i >= max_seq_len:
        #     break
        for letter in word:
            label_encoded[i] = alphabet_dict[letter]
            i += 1
    return image, label_encoded

def decode(prediction):
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

def run_epoch(e, network, dataloader, trainer, log_dir, print_name, update_network, save_network):
    total_losses = [nd.zeros(1, ctx_i) for ctx_i in ctx]
    for i, (X, Y) in enumerate(dataloader):
        X = gluon.utils.split_and_load(X, ctx)
        Y = gluon.utils.split_and_load(Y, ctx)

        with autograd.record():
            losses = []
            for x, y in zip(X, Y):
                output = network(x)
                loss_ctc = ctc_loss(output, y)
                loss_ctc = (y != -1).sum(axis=1)*loss_ctc
                losses.append(loss_ctc)

        if update_network:
            for loss in losses:
                loss.backward()
        if i == 0 and e % send_image_every_n == 0 and e > 0:
            predictions = output.softmax().topk(axis=2).asnumpy()
            decoded_text = decode(predictions)
            output_image = draw_text_on_image(x.asnumpy(), decoded_text)
            with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
                sw.add_image('bb_{}_image'.format(print_name), output_image, global_step=e)

        for index, loss in enumerate(losses):
            total_losses[index] += loss.mean()/len(ctx)

        step_size = 0
        for x in X:
            step_size += x.shape[0]
        trainer.step(step_size)
        
    total_loss = 0
    for loss in total_losses:
        total_loss += loss.asscalar()
    epoch_loss = float(total_loss)/len(dataloader)

    with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
        sw.add_scalar('loss', {print_name: epoch_loss}, global_step=e)

    if save_network and e % save_every_n == 0 and e > 0:
        network.save_parameters("{}/{}".format(checkpoint_dir, checkpoint_name))

    return epoch_loss
    
if __name__ == "__main__":
    gpu_count = 4
    ctx = [mx.gpu(i) for i in range(gpu_count)]
    expand_bb_scale = 0.05
    
    epochs = 500
    learning_rate = 0.0001
    batch_size = 32 * len(ctx)

    random_y_translation, random_x_translation = 0.03, 0.03

    log_dir = "./logs"
    checkpoint_dir = "model_checkpoint"
    checkpoint_name = "handwriting.params"

    train_ds = IAMDataset("line", output_data="text", train=True)
    print("Number of training samples: {}".format(len(train_ds)))

    test_ds = IAMDataset("line", output_data="text", train=False)
    print("Number of testing samples: {}".format(len(test_ds)))

    train_data = gluon.data.DataLoader(train_ds.transform(transform), batch_size, shuffle=True, last_batch="discard", num_workers=multiprocessing.cpu_count()-2)
    test_data = gluon.data.DataLoader(test_ds.transform(transform), batch_size, shuffle=False, last_batch="discard", num_workers=multiprocessing.cpu_count()-2)

    net = Network()
    net.hybridize()

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, })
    
    ctc_loss = gluon.loss.CTCLoss(weight=0.2)

    for e in range(epochs):
        train_loss = run_epoch(e, net, train_data, trainer, log_dir, print_name="train", 
                               update_network=True, save_network=True)
        test_loss = run_epoch(e, net, test_data, trainer, log_dir, print_name="test", 
                              update_network=False, save_network=False)
        if e % print_every_n == 0: # and e > 0:
            print("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}".format(e, train_loss, test_loss))
