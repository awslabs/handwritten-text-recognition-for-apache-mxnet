import multiprocessing
import time
import random
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import mxnet as mx
import numpy as np
from skimage.draw import line_aa
from skimage import transform as tf

from mxnet import nd, autograd, gluon
from mxnet.image import resize_short
from mxboard import SummaryWriter

ctx = mx.gpu()
mx.random.seed(1)

from utils.iam_dataset import IAMDataset
from utils.iou_loss import IOU_loss
from utils.draw_box_on_image import draw_box_on_image

MODEL_CHECKPOINT_FOLDER = "model_checkpoint"
if not os.path.isdir(MODEL_CHECKPOINT_FOLDER):
    os.makedirs(MODEL_CHECKPOINT_FOLDER)

train_ds = IAMDataset("form", output_data="bb", output_parse_method="form", train=True)
print("Number of training samples: {}".format(len(train_ds)))

test_ds = IAMDataset("form", output_data="bb", output_parse_method="form", train=False)
print("Number of testing samples: {}".format(len(test_ds)))

batch_size = 32

def transform(data, label):
    image = mx.nd.array(data).expand_dims(axis=2)
    image = resize_short(image, int(800/3))
    image = image.transpose([2, 0, 1])/255.
    label = label[0].astype(np.float32)
    return image, mx.nd.array(label)

def augment_transform(data, label):
    width_range = 0.05
    height_range = 0.05

    ty = random.uniform(-height_range, height_range)
    tx = random.uniform(-width_range, width_range)
    st = tf.SimilarityTransform(translation=(tx*data.shape[1], ty*data.shape[0]))
    data = tf.warp(data, st)

    label[0][0] = label[0][0] - tx
    label[0][1] = label[0][1] - ty
    return transform(data*255., label)
    
train_data = gluon.data.DataLoader(train_ds.transform(augment_transform), batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
test_data = gluon.data.DataLoader(test_ds.transform(transform), batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())

def make_cnn():
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
    return cnn

cnn = make_cnn()
cnn.load_params("model_checkpoint/cnn300.params", ctx=ctx)

iou_loss = IOU_loss()
LEARNING_RATE = 0.0001
trainer = gluon.Trainer(cnn.collect_params(), 'adam', {'learning_rate': LEARNING_RATE, })

epochs = 500
print_every_n = 20
send_image_every_n = 20
save_every_n = 100

for e in range(epochs):    
    loss = nd.zeros(1, ctx)
    test_loss = nd.zeros(1, ctx)
    tick = time.time()
    acc = nd.zeros(1, ctx)
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        
        with autograd.record():
            output = cnn(data)
            loss_iou = iou_loss(output, label)
        loss_iou.backward()
        loss += loss_iou.mean()
        
        trainer.step(data.shape[0])
        if e % send_image_every_n == 0 and e > 0 and i == 0:
            output = cnn(data)
            data_np = data.asnumpy()
            label_np = label.asnumpy()
            pred_np = output.asnumpy()
            train_output_image = draw_box_on_image(pred_np, label_np, data_np)

    for i, (data, label) in enumerate(test_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = cnn(data)
            test_loss_iou = iou_loss(output, label)
        test_loss += test_loss_iou.mean()
        
        # Generate images of the first batch
        if e % send_image_every_n == 0 and e > 0 and i == 0:
            output = cnn(data)
            data_np = data.asnumpy()
            label_np = label.asnumpy()
            pred_np = output.asnumpy()
            test_output_image = draw_box_on_image(pred_np, label_np, data_np)

    train_loss = float(loss.asscalar())/len(train_data)
    test_loss = float(test_loss.asscalar())/len(test_data)
    
    if e % print_every_n == 0 and e > 0:
        print("Epoch {0}, train_loss {1:.6f}, val_loss {2:.6f}".format(
            e, train_loss, test_loss))
    
    with SummaryWriter(logdir='./logs', verbose=False, flush_secs=5) as sw:
        sw.add_scalar('loss', {"train": train_loss, "test": test_loss}, global_step=e)
        if e % send_image_every_n == 0 and e > 0:
            train_output_image[train_output_image<0] = 0
            train_output_image[train_output_image>1] = 1

            test_output_image[test_output_image<0] = 0
            test_output_image[test_output_image>1] = 1
            sw.add_image('bb_train_image', train_output_image, global_step=e)
            sw.add_image('bb_test_image', test_output_image, global_step=e)

    if e % save_every_n == 0 and e > 0:
        cnn.save_params("model_checkpoint/cnn_iou{}.params".format(e))
