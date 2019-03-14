# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import multiprocessing
import time
import random
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import mxnet as mx
import numpy as np
from skimage.draw import line_aa
from skimage import transform as skimage_transform

from mxnet import nd, autograd, gluon
from mxnet.image import resize_short
from mxboard import SummaryWriter

mx.random.seed(1)

from .utils.iam_dataset import IAMDataset, resize_image
from .utils.iou_loss import IOU_loss
from .utils.draw_box_on_image import draw_box_on_image

print_every_n = 20
send_image_every_n = 20
save_every_n = 100

# pre-training: python paragraph_segmentation_dcnn.py -r 0.001 -e 181 -n cnn_mse.params -y 0.15
# fine-tuning: python paragraph_segmentation_dcnn.py -r 0.0001 -l iou -e 150 -n cnn_iou.params -f cnn_mse.params -x 0 -y 0

def paragraph_segmentation_transform(image, image_size):
    '''
    Function used for inference to resize the image for paragraph segmentation
    '''
    resized_image, _ = resize_image(image, image_size)
    
    resized_image = mx.nd.array(resized_image).expand_dims(axis=2)
    resized_image = mx.image.resize_short(resized_image, int(800/3))
    resized_image = resized_image.transpose([2, 0, 1])/255.
    resized_image = resized_image.expand_dims(axis=0)
    return resized_image

def transform(data, label):
    '''
    Function that converts "data"" into the input image tensor for a CNN
    Label is converted into a float tensor.
    '''
    image = mx.nd.array(data).expand_dims(axis=2)
    image = resize_short(image, int(800/3))
    image = image.transpose([2, 0, 1])/255.
    label = label[0].astype(np.float32)
    
    bb = label.copy()
    new_w = (1 + expand_bb_scale) * bb[2]
    new_h = (1 + expand_bb_scale) * bb[3]
    
    bb[0] = bb[0] - (new_w - bb[2])/2
    bb[1] = bb[1] - (new_h - bb[3])/2
    bb[2] = new_w
    bb[3] = new_h

    return image, mx.nd.array(bb)

def augment_transform(data, label):
    '''
    Function that randomly translates the input image by +-width_range and +-height_range.
    The labels (bounding boxes) are also translated by the same amount.
    '''
    ty = random.uniform(-random_y_translation, random_y_translation)
    tx = random.uniform(-random_x_translation, random_x_translation)
    st = skimage_transform.SimilarityTransform(translation=(tx*data.shape[1], ty*data.shape[0]))
    data = skimage_transform.warp(data, st)
    label = label.copy()
    label[0][0] = label[0][0] - tx
    label[0][1] = label[0][1] - ty
    return transform(data*255., label)

class SegmentationNetwork(gluon.nn.HybridBlock):
    
    def __init__(self, p_dropout = 0.5, ctx=mx.cpu()):
        super(SegmentationNetwork, self).__init__()

        pretrained = gluon.model_zoo.vision.resnet34_v1(pretrained=True, ctx=ctx)
        first_weights = pretrained.features[0].weight.data().mean(axis=1).expand_dims(axis=1)

        body = gluon.nn.HybridSequential(prefix="SegmentationNetwork_")
        with body.name_scope():
            first_layer = gluon.nn.Conv2D(channels=64, kernel_size=(7, 7), padding=(3, 3), strides=(2, 2), in_channels=1, use_bias=False)
            first_layer.initialize(mx.init.Normal(), ctx=ctx)
            first_layer.weight.set_data(first_weights)
            body.add(first_layer)
            body.add(*pretrained.features[1:6])
        
            output = gluon.nn.HybridSequential()
            with output.name_scope():
                output.add(gluon.nn.Flatten())
                output.add(gluon.nn.Dense(64, activation='relu'))
                output.add(gluon.nn.Dropout(p_dropout))
                output.add(gluon.nn.Dense(64, activation='relu'))
                output.add(gluon.nn.Dropout(p_dropout))
                output.add(gluon.nn.Dense(4, activation='sigmoid'))

            output.collect_params().initialize(mx.init.Normal(), ctx=ctx)
            body.add(output)
        self.cnn = body

    def hybrid_forward(self, F, x):
        return self.cnn(x)
    
def make_cnn_old():
    p_dropout = 0.5

    cnn = gluon.nn.HybridSequential()
    cnn.add(gluon.nn.Conv2D(kernel_size=(3,3), padding=(1,1), channels=16, activation="relu"))
    cnn.add(gluon.nn.BatchNorm())

    cnn.add(gluon.nn.Conv2D(kernel_size=(3,3), padding=(1,1), channels=16, activation="relu"))
    cnn.add(gluon.nn.BatchNorm())

    cnn.add(gluon.nn.Conv2D(kernel_size=(3,3), padding=(1,1), channels=16, activation="relu"))
    cnn.add(gluon.nn.BatchNorm())

    cnn.add(gluon.nn.Conv2D(kernel_size=(3,3), padding=(1,1), channels=16, activation="relu"))
    cnn.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    cnn.add(gluon.nn.BatchNorm())

    cnn.add(gluon.nn.Conv2D(kernel_size=(3,3), padding=(1,1), channels=16, activation="relu"))
    cnn.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    cnn.add(gluon.nn.BatchNorm())

    cnn.add(gluon.nn.Flatten())
    cnn.add(gluon.nn.Dense(64, activation='relu'))
    cnn.add(gluon.nn.Dropout(p_dropout))
    cnn.add(gluon.nn.Dense(64, activation='relu'))
    cnn.add(gluon.nn.Dropout(p_dropout))
    cnn.add(gluon.nn.Dense(4, activation='sigmoid'))

    cnn.hybridize()
    cnn.collect_params().initialize(mx.init.Normal(), ctx=ctx)
    return cnn

def run_epoch(e, network, dataloader, loss_function, trainer, log_dir, print_name, is_train):
    total_loss = nd.zeros(1, ctx)
    for i, (data, label) in enumerate(dataloader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        
        with autograd.record(train_mode=is_train):
            output = network(data)
            loss_i = loss_function(output, label)
        if is_train:
            loss_i.backward()
            trainer.step(data.shape[0])

        total_loss += loss_i.mean()
        
        if e % send_image_every_n == 0 and e > 0 and i == 0:
            output_image = draw_box_on_image(output.asnumpy(), label.asnumpy(), data.asnumpy())
    epoch_loss = float(total_loss .asscalar())/len(dataloader)
    
    with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
        sw.add_scalar('loss', {print_name: epoch_loss}, global_step=e)
        if e % send_image_every_n == 0 and e > 0:
            output_image[output_image<0] = 0
            output_image[output_image>1] = 1
            sw.add_image('bb_{}_image'.format(print_name), output_image, global_step=e)
            
    return epoch_loss

def main(ctx=mx.gpu()):
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_ds = IAMDataset("form", output_data="bb", output_parse_method="form", train=True)
    print("Number of training samples: {}".format(len(train_ds)))

    test_ds = IAMDataset("form", output_data="bb", output_parse_method="form", train=False)
    print("Number of testing samples: {}".format(len(test_ds)))

    train_data = gluon.data.DataLoader(train_ds.transform(augment_transform), batch_size,
                                       shuffle=True, num_workers=int(multiprocessing.cpu_count()/2))
    test_data = gluon.data.DataLoader(test_ds.transform(transform), batch_size,
                                      shuffle=False, num_workers=int(multiprocessing.cpu_count()/2))

    net = SegmentationNetwork()
    net.hybridize()
    net.collect_params().reset_ctx(ctx)
    if restore_checkpoint_name:
        net.load_parameters("{}/{}".format(checkpoint_dir, restore_checkpoint_name), ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})
    best_test_loss = 10e5
    for e in range(epochs):
        train_loss = run_epoch(e, net, train_data, loss_function=loss_function, log_dir=log_dir, 
                               trainer=trainer, print_name="train", is_train=True)
        test_loss = run_epoch(e, net, test_data, loss_function=loss_function, log_dir=log_dir,
                              trainer=trainer, print_name="test", is_train=False)
        if test_loss < best_test_loss:
            print("Saving network, previous best test loss {:.6f}, current test loss {:.6f}".format(best_test_loss, test_loss))
            net.save_parameters(os.path.join(checkpoint_dir, checkpoint_name))
            best_test_loss = test_loss
        if e % print_every_n == 0 and e > 0:
            print("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}".format(e, train_loss, test_loss))

if __name__ == "__main__":
    loss_options = ["mse", "iou"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", default=0,
                        help="Gpu ID to use, -1 CPU")
    parser.add_argument("-l", "--loss", default="mse",
                        help="Set loss function of the network. Options: {}".format(loss_options))
    parser.add_argument("-e", "--epochs", default=300,
                        help="The number of epochs to run")
    parser.add_argument("-b", "--batch_size", default=32,
                        help="The batch size used for training")
    parser.add_argument("-r", "--learning_rate", default=0.001,
                        help="The learning rate used for training")

    parser.add_argument("-c", "--checkpoint_dir", default="model_checkpoint",
                        help="Directory name for the model checkpoint")
    parser.add_argument("-n", "--checkpoint_name", default="cnn.params",
                        help="Name for the model checkpoint")
    parser.add_argument("-f", "--restore_checkpoint_name", default=None,
                        help="Name for the model to restore from")

    parser.add_argument("-d", "--log_dir", default="./logs",
                        help="Location to save the MXBoard logs")

    parser.add_argument("-s", "--expand_bb_scale", default=0.03,
                        help="Scale to expand the bounding box")
    parser.add_argument("-x", "--random_x_translation", default=0.05,
                        help="Randomly translate the image by x%")
    parser.add_argument("-y", "--random_y_translation", default=0.05,
                        help="Randomly translate the image by y%")
    
    args = parser.parse_args()
    print(args)
    loss = args.loss
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    checkpoint_dir = args.checkpoint_dir
    checkpoint_name = args.checkpoint_name
    restore_checkpoint_name = args.restore_checkpoint_name
    log_dir = args.log_dir
    expand_bb_scale = float(args.expand_bb_scale)
    random_x_translation = float(args.random_x_translation)
    random_y_translation = float(args.random_y_translation)
    gpu_id = int(args.gpu_id)
    ctx = mx.gpu(gpu_id) if gpu_id != -1 else mx.cpu()

    assert loss in loss_options, "{} is not an available option {}".format(loss, loss_options)

    if loss == "iou":
        loss_function = IOU_loss()
    elif loss == "mse":
        loss_function = gluon.loss.L2Loss()

    if restore_checkpoint_name:
        restore_checkpoint = os.path.join(checkpoint_dir, restore_checkpoint_name)
        assert os.path.isfile(restore_checkpoint), "{} does not exist".format(os.path.join(checkpoint_dir, restore_checkpoint_name))
    main(ctx)
