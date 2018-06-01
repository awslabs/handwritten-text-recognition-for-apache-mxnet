import multiprocessing
import time
import random
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import mxnet as mx
import numpy as np
from skimage.draw import line_aa

from mxnet import nd, autograd, gluon
from mxnet.image import resize_short
from mxboard import SummaryWriter

ctx = mx.gpu()
mx.random.seed(1)

from utils.iam_dataset import IAMDataset

MODEL_CHECKPOINT_FOLDER = "model_checkpoint"
if not os.path.isdir(MODEL_CHECKPOINT_FOLDER):
    os.makedirs(MODEL_CHECKPOINT_FOLDER)

train_ds = IAMDataset("form", output_data="bb", output_parse_method="form", train=True)
print("Number of training samples: {}".format(len(train_ds)))

test_ds = IAMDataset("form", output_data="bb", output_parse_method="form", train=False)
print("Number of testing samples: {}".format(len(test_ds)))

batch_size = 32

def tf(data, label):
    image = mx.nd.array(data).expand_dims(axis=2)
    image = resize_short(image, int(800/3))
    image = image.transpose([2, 0, 1])/255.
    label = label[0].astype(np.float32)

    return image, mx.nd.array(label)

train_data = gluon.data.DataLoader(train_ds.transform(tf), batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
test_data = gluon.data.DataLoader(test_ds.transform(tf), batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())

p_dropout = 0.5

def make_cnn():
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
cnn = make_cnn()

LEARNING_RATE = 0.001
l2_loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(cnn.collect_params(), 'adam', {'learning_rate': LEARNING_RATE, })

def draw_box_on_image(output, data):    
    image_h, image_w = data.shape[-2:]
    output[:, 0], output[:, 1] = output[:, 0] * image_w, output[:, 1] * image_h
    output[:, 2], output[:, 3] = output[:, 2] * image_w, output[:, 3] * image_h

    for i in range(data.shape[0]):
        image = data[i, 0]

        (x, y, w, h) = output[i, :]
        (x1, y1, x2, y2) = (x, y, x + w, y + h)
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
        
        rr, cc, val = line_aa(y1, x1, y2, x1)
        image[rr, cc] = 0
        
        rr, cc, val = line_aa(y2, x1, y2, x2)
        image[rr, cc] = 0
        
        rr, cc, val = line_aa(y2, x2, y1, x2)
        image[rr, cc] = 0
        
        rr, cc, val = line_aa(y1, x2, y1, x1)
        image[rr, cc] = 0

        data[i, 0, rr, cc] = image[rr, cc]

    return data

epochs = 5000
print_every_n = 20
send_image_every_n = 100
lowest_loss = 1e5

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
            loss_l2 = l2_loss(output, label)
        loss_l2.backward()
        loss += loss_l2.mean()
        
        trainer.step(data.shape[0])
    
    for i, (data, label) in enumerate(test_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = cnn(data)
            test_loss_l2 = l2_loss(output, label)
        test_loss += test_loss_l2.mean()
        
        # Generate images of the first batch
        if e % send_image_every_n == 0 and e > 0 and i == 0:
            output = cnn(data)
            data_np = data.asnumpy()
            output_np = output.asnumpy()
            output_image = draw_box_on_image(output_np, data_np)

    train_loss = float(loss.asscalar())/len(train_data)
    test_loss = float(test_loss.asscalar())/len(test_data)
    if test_loss < lowest_loss:
        cnn.save_params(os.path.join(MODEL_CHECKPOINT_FOLDER, "tmp.glu"))
    
    if e % print_every_n == 0 and e > 0:
        print("Epoch {0}, train_loss {1:.6f}, val_loss {2:.6f}".format(
            e, train_loss, test_loss))
    
    with SummaryWriter(logdir='./logs', verbose=False, flush_secs=5) as sw:
        sw.add_scalar('loss', {"train": train_loss, "test": test_loss}, global_step=e)
        if e % send_image_every_n == 0 and e > 0:
            output_image[output_image<0] = 0
            output_image[output_image>1] = 1
            sw.add_image('bb_image', output_image, global_step=e)
