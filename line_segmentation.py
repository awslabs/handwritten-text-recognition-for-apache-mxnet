import multiprocessing
import time
import random
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import mxnet as mx
from mxnet.contrib.ndarray import MultiBoxPrior, MultiBoxTarget, MultiBoxDetection
import numpy as np
from skimage.draw import line_aa
from skimage import transform as skimage_tf

from mxnet import nd, autograd, gluon
from mxnet.image import resize_short
from mxboard import SummaryWriter

ctx = mx.gpu()
mx.random.seed(1)

from utils.iam_dataset import IAMDataset
from utils.draw_box_on_image import draw_boxes_on_image

batch_size = 32
min_c = 0.5

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

    cnn.hybridize()
    cnn.collect_params().initialize(mx.init.Normal(), ctx=ctx)
    return cnn

def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return gluon.nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return gluon.nn.Conv2D(num_anchors * 4, 3, padding=1)

def down_sample(num_filters):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer
    to halve the feature size"""
    out = gluon.nn.HybridSequential()
    for _ in range(2):
        out.add(gluon.nn.Conv2D(num_filters, 3, strides=1, padding=1))
        out.add(gluon.nn.BatchNorm(in_channels=num_filters))
        out.add(gluon.nn.Activation('relu'))
    out.add(gluon.nn.MaxPool2D(2))
    return out

body = make_cnn()

def flatten_prediction(pred):
    return nd.flatten(nd.transpose(pred, axes=(0, 2, 3, 1)))

def concat_predictions(preds):
    return nd.concat(*preds, dim=1)

def ssd_model(num_anchors, num_classes):
    downsamples = gluon.nn.Sequential()
    class_preds = gluon.nn.Sequential()
    box_preds = gluon.nn.Sequential()

    downsamples.add(down_sample(128))
    downsamples.add(down_sample(128))
    downsamples.add(down_sample(128))

    for scale in range(5):
        class_preds.add(class_predictor(num_anchors, num_classes))
        box_preds.add(box_predictor(num_anchors))

    return body, downsamples, class_preds, box_preds

def ssd_forward(x, body, downsamples, class_preds, box_preds, sizes, ratios):
    x = body(x)

    default_anchors = []
    predicted_boxes = []
    predicted_classes = []

    for i in range(5):
        default_anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))
        predicted_boxes.append(flatten_prediction(box_preds[i](x)))
        predicted_classes.append(flatten_prediction(class_preds[i](x)))
        if i < 3:
            x = downsamples[i](x)
        elif i == 3:
            x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))

    return default_anchors, predicted_classes, predicted_boxes

class SSD(gluon.Block):
    def __init__(self, num_classes, **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.anchor_sizes = [[.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        self.anchor_ratios = [[1, 2, .5]] * 5
        self.num_classes = num_classes

        with self.name_scope():
            self.body, self.downsamples, self.class_preds, self.box_preds = ssd_model(4, num_classes)
            self.body.initialize(mx.init.Normal(), ctx=ctx)
            self.downsamples.initialize(mx.init.Normal(), ctx=ctx)
            self.class_preds.initialize(mx.init.Normal(), ctx=ctx)
            self.box_preds.initialize(mx.init.Normal(), ctx=ctx)

    def forward(self, x):
        default_anchors, predicted_classes, predicted_boxes = ssd_forward(x, self.body, self.downsamples,
            self.class_preds, self.box_preds, self.anchor_sizes, self.anchor_ratios)
        # we want to concatenate anchors, class predictions, box predictions from different layers
        anchors = concat_predictions(default_anchors)
        box_preds = concat_predictions(predicted_boxes)
        class_preds = concat_predictions(predicted_classes)
        class_preds = nd.reshape(class_preds, shape=(0, -1, self.num_classes + 1))
        return anchors, class_preds, box_preds
    
def training_targets(default_anchors, class_predicts, labels):
    class_predicts = nd.transpose(class_predicts, axes=(0, 2, 1))
    box_target, box_mask, cls_target = MultiBoxTarget(default_anchors, labels, class_predicts)
    return box_target, box_mask, cls_target

def transform(image, label):
    desired_image_size = (700, 700)
    max_label_n = 13
    '''
    Function that converts "data"" into the input image tensor for a CNN
    Label is converted into a float tensor.
    '''

    # Preprocess image
    size = image.shape[:2]
    ratio_w = float(desired_image_size[0])/size[0]
    ratio_h = float(desired_image_size[1])/size[1]
    ratio = min(ratio_w, ratio_h)
    new_size = tuple([int(x*ratio) for x in size])
    image = skimage_tf.resize(image, (new_size[1], new_size[0]))
    size = image.shape
            
    delta_w = max(0, desired_image_size[1] - size[1])
    delta_h = max(0, desired_image_size[0] - size[0])
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
            
    color = image[0][0]
    if color < 230:
        color = 230
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))
    
    image = np.expand_dims(image, axis=2)
    image = mx.nd.array(image)
    image = resize_short(image, int(700/3))
    image = image.transpose([2, 0, 1])
    
    # Preprocess label
    
    # Expand the bounding box to relax the boundaries
    # expand_bb_scale = 0.03
    
    # bb = label

    # new_w = (1 + expand_bb_scale) * bb[:, 2]
    # new_h = (1 + expand_bb_scale) * bb[:, 3]

    # bb[:, 0] = bb[:, 0] - (new_w - bb[:, 2])/2
    # bb[:, 1] = bb[:, 1] - (new_h - bb[:, 3])/2
    # bb[:, 2] = new_w
    # bb[:, 3] = new_h
    label = label.astype(np.float32)
    label_n = label.shape[0]
    label_padded = np.zeros(shape=(max_label_n, 5))
    label_padded[:label_n, 1:] = label
    label_padded[:label_n, 0] = np.ones(shape=(1, label_n))
    label_padded = mx.nd.array(label_padded)
    return image, label_padded

train_ds = IAMDataset("form_bb", output_data="bb", output_parse_method="line", train=True)
print("Number of training samples: {}".format(len(train_ds)))

test_ds = IAMDataset("form_bb", output_data="bb", output_parse_method="line", train=False)
print("Number of testing samples: {}".format(len(test_ds)))

train_data = gluon.data.DataLoader(train_ds.transform(transform), batch_size, shuffle=True)#, num_workers=multiprocessing.cpu_count())
test_data = gluon.data.DataLoader(test_ds.transform(transform), batch_size, shuffle=False)#, num_workers=multiprocessing.cpu_count())

learning_rate = 0.0001
epochs = 150

net = SSD(2)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, })

cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return F.mean(loss, self._batch_axis, exclude=True)

box_loss = SmoothL1Loss()
cls_metric = mx.metric.Accuracy()
box_metric = mx.metric.MAE()

print_every_n = 5
send_image_every_n = 5
save_every_n = 10
checkpoint_dir, checkpoint_name = "model_checkpoint", "ssd.params"

def run_epoch(e, network, dataloader, trainer, log_dir, print_name, update_cnn, update_metric, save_cnn):
    total_loss = nd.zeros(1, ctx)
    for i, (x, y) in enumerate(dataloader):
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)
        
        with autograd.record():
            default_anchors, class_predictions, box_predictions = network(x)
            box_target, box_mask, cls_target = training_targets(default_anchors, class_predictions, y)
            # losses
            loss1 = cls_loss(class_predictions, cls_target)
            loss2 = box_loss(box_predictions, box_target, box_mask)
            # sum all losses
            loss = loss1 + loss2
            
        if update_cnn:
            loss.backward()
            trainer.step(x.shape[0])

        total_loss += loss.mean()
        if update_metric:
            cls_metric.update([cls_target], [nd.transpose(class_predictions, (0, 2, 1))])
            box_metric.update([box_target], [box_predictions * box_mask])
        if i == 0 and e % send_image_every_n == 0 and e > 0:
            cls_probs = nd.SoftmaxActivation(nd.transpose(class_predictions, (0, 2, 1)), mode='channel')
            output = MultiBoxDetection(*[cls_probs, box_predictions, default_anchors], force_suppress=True, clip=False)
            output = output.asnumpy()

            predicted_bb = []
            for b in range(output.shape[0]):
                predicted_bb_ = output[b, output[b, :, 0] != -1]

                b_higher_than_min_c = predicted_bb_[:, 1] > min_c
                # filter all class predictions lower than min_c and remove the class prediction confidences
                predicted_bb_ = predicted_bb_[b_higher_than_min_c, 2:]  
                predicted_bb.append(predicted_bb_)
            output_image = draw_boxes_on_image(predicted_bb, y[:, :, 1:].asnumpy(), x.asnumpy())

    epoch_loss = float(total_loss.asscalar())/len(dataloader)

    with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
        if update_metric:
            name1, val1 = cls_metric.get()
            name2, val2 = box_metric.get()
            sw.add_scalar(name1, {"test": val1}, global_step=e)
            sw.add_scalar(name2, {"test": val2}, global_step=e)

        sw.add_scalar('loss', {print_name: epoch_loss}, global_step=e)
        if e % send_image_every_n == 0 and e > 0:
            output_image[output_image<0] = 0
            output_image[output_image>1] = 1
            sw.add_image('bb_{}_image'.format(print_name), output_image, global_step=e)
            
    if save_cnn and e % save_every_n == 0 and e > 0:
        network.save_params("{}/{}".format(checkpoint_dir, checkpoint_name))
    return epoch_loss

for e in range(epochs):
    cls_metric.reset()
    box_metric.reset()

    log_dir = "./logs"
    train_loss = run_epoch(e, net, train_data, trainer, log_dir, print_name="train", update_cnn=True, update_metric=False, save_cnn=False)
    test_loss = run_epoch(e, net, test_data, trainer, log_dir, print_name="test", update_cnn=False, update_metric=True, save_cnn=False)
    if e % print_every_n == 0:
        name1, val1 = cls_metric.get()
        name2, val2 = box_metric.get()
        print("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}, test {3}={4:.6f}, {5}={6:.6f}".format(e, train_loss, test_loss,
                                                                                           name1, val1, name2, val2))
