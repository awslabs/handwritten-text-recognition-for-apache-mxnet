import time
import random
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import mxnet as mx
from mxnet.contrib.ndarray import MultiBoxPrior, MultiBoxTarget, MultiBoxDetection, box_nms
import numpy as np
from skimage.draw import line_aa
from skimage import transform as skimage_tf

from mxnet import nd, autograd, gluon
from mxnet.image import resize_short
from mxboard import SummaryWriter
from mxnet.gluon.model_zoo.vision import resnet34_v1
np.seterr(all='raise')

import multiprocessing

GPU_COUNT = 4
ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
mx.random.seed(1)

from utils.iam_dataset import IAMDataset
from utils.draw_box_on_image import draw_boxes_on_image

expand_bb_scale = 0.03
min_c = 0.01
overlap_thres = 0.20
topk = 13
epochs = 351
learning_rate = 0.0001

# learning_rate = 0.000001
# maximum is 1% otherwise the words will be outside the box
random_y_translation, random_x_translation = 0.01, 0.01
random_remove_box = 0.15

random_y_translation, random_x_translation = 0.00, 0.00
random_remove_box = 0.00

batch_size = 32 * len(ctx)
print_every_n = 5
send_image_every_n = 20
save_every_n = 50
checkpoint_dir, checkpoint_name = "model_checkpoint", "ssd.params"

def make_cnn():    
    pretrained = resnet34_v1(pretrained=True, ctx=ctx)
    pretrained_2 = resnet34_v1(pretrained=True, ctx=mx.cpu(0))
    first_weights = pretrained_2.features[0].weight.data().mean(axis=1).expand_dims(axis=1)
    
    body = gluon.nn.HybridSequential()
    with body.name_scope():
        first_layer = gluon.nn.Conv2D(channels=64, kernel_size=(7, 7), padding=(3, 3), strides=(2, 2), in_channels=1, use_bias=False)
        first_layer.initialize(mx.init.Normal(), ctx=ctx)
        first_layer.weight.set_data(first_weights)
        body.add(first_layer)
        body.add(*pretrained.features[1:-3])
        body.hybridize()
    return body

def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return gluon.nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

def box_predictor(num_anchors):
    pred = gluon.nn.HybridSequential()
    with pred.name_scope():
        pred.add(gluon.nn.Conv2D(channels=num_anchors * 4, kernel_size=(3, 3), padding=1))
        pred.add(gluon.nn.BatchNorm())

        pred.add(gluon.nn.Conv2D(channels=num_anchors * 4, kernel_size=3, padding=1))
        pred.add(gluon.nn.BatchNorm())

        pred.add(gluon.nn.Conv2D(channels=num_anchors * 4, kernel_size=3, padding=1))
        pred.add(gluon.nn.BatchNorm())

        pred.add(gluon.nn.Conv2D(channels=num_anchors * 4, kernel_size=3, padding=1))

    pred.hybridize()
    return pred

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
        self.anchor_sizes = [[.75, .79], [.79, .84], [.81, .85], [.85, .89], [.88, .961]] #TODO: maybe reduce the number of types boxes?
        self.anchor_ratios = [[10, 8, 6], [9, 7, 5], [7, 5, 3], [6, 4, 2], [5, 3, 1]] 
        self.num_classes = num_classes

        with self.name_scope():
            self.body, self.downsamples, self.class_preds, self.box_preds = ssd_model(4, num_classes)
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

def augment_transform(data, label):
    '''
    1) Function that randomly translates the input image by +-width_range and +-height_range.
    The labels (bounding boxes) are also translated by the same amount.
    2) Each line can also be randomly removed for augmentation. Labels are also reduced to correspond to this
    data and label are converted into tensors by calling the "transform" function.
    '''
    ty = random.uniform(-random_y_translation, random_y_translation)
    tx = random.uniform(-random_x_translation, random_x_translation)
    st = skimage_tf.SimilarityTransform(translation=(tx*data.shape[1], ty*data.shape[0]))
    data = skimage_tf.warp(data, st, cval=1.0)

    label[:, 0] = label[:, 0] - tx/2 #NOTE: Check why it has to be halfed (found experimentally)
    label[:, 1] = label[:, 1] - ty/2
    
    index = np.random.uniform(0, 1.0, size=label.shape[0]) > random_remove_box
    for i, should_output_bb in enumerate(index):
        if should_output_bb == False:
            (x, y, w, h) = label[i]
            (x1, y1, x2, y2) = (x, y, x + w, y + h)
            (x1, y1, x2, y2) = (x1 * data.shape[1], y1 * data.shape[0],
                                x2 * data.shape[1], y2 * data.shape[0])
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            x1 = 0 if x1 < 0 else x1
            y1 = 0 if y1 < 0 else y1
            image_h, image_w = data.shape
            x2 = image_w if x2 > image_w else x2
            y2 = image_h if y2 > image_h else y2

            mean_value = 1.0 #np.mean(data[y1:y2, x1:x2])
            data[y1:y2, x1:x2] = mean_value
                
    augmented_labels = label[index, :]
    return transform(data*255., augmented_labels)

def transform(image, label):
    '''
    Function that converts "data"" into the input image tensor for a CNN
    Label is converted into a float tensor.
    '''
    max_label_n = 13

    # Resize the image
    image = np.expand_dims(image, axis=2)
    image = mx.nd.array(image)
    image = resize_short(image, int(700/2))
    image = image.transpose([2, 0, 1])/255.

    # Expand the bounding box by expand_bb_scale
    bb = label
    new_w = (1 + expand_bb_scale) * bb[:, 2]
    new_h = (1 + expand_bb_scale) * bb[:, 3]
    
    bb[:, 0] = bb[:, 0] - (new_w - bb[:, 2])/2
    bb[:, 1] = bb[:, 1] - (new_h - bb[:, 3])/2
    bb[:, 2] = new_w
    bb[:, 3] = new_h
    label = bb 

    # Convert the predicted bounding box from (x, y, w, h to (x, y, x + w, y + h)
    label = label.astype(np.float32)
    label[:, 2] = label[:, 0] + label[:, 2]
    label[:, 3] = label[:, 1] + label[:, 3]

    # Zero pad the data
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

train_data = gluon.data.DataLoader(train_ds.transform(augment_transform), batch_size, shuffle=True, last_batch="discard", num_workers=multiprocessing.cpu_count()-2)
test_data = gluon.data.DataLoader(test_ds.transform(transform), batch_size, shuffle=False, last_batch="discard", num_workers=multiprocessing.cpu_count()-2)

net = SSD(2)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, })

# data, label = train_ds[0]
# data, _ = transform(data, label)
# data = data.expand_dims(axis=0)
# data = data.as_in_context(ctx)
# net.summary(data)
# import pdb; pdb.set_trace();

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

def run_epoch(e, network, dataloader, trainer, log_dir, print_name, update_cnn, update_metric, save_cnn):
    total_losses = [nd.zeros(1, ctx_i) for ctx_i in ctx]
    for i, (X, Y) in enumerate(dataloader):
        X = gluon.utils.split_and_load(X, ctx)
        Y = gluon.utils.split_and_load(Y, ctx)
        
        with autograd.record():
            losses = []
            for x, y in zip(X, Y):
                default_anchors, class_predictions, box_predictions = network(x)
                box_target, box_mask, cls_target = training_targets(default_anchors, class_predictions, y)
                # losses
                loss_class = cls_loss(class_predictions, cls_target)
                loss_box = box_loss(box_predictions, box_target, box_mask)
                # sum all losses
                loss = loss_class + loss_box
                losses.append(loss)
            
        if update_cnn:
            for loss in losses:
                loss.backward()
            step_size = 0
            for x in X:
                step_size += x.shape[0]
            trainer.step(step_size)

        for index, loss in enumerate(losses):
            total_losses[index] += loss.mean()/len(ctx)
            
        if update_metric:
            cls_metric.update([cls_target], [nd.transpose(class_predictions, (0, 2, 1))])
            box_metric.update([box_target], [box_predictions * box_mask])
            
        if i == 0 and e % send_image_every_n == 0 and e > 0:
            cls_probs = nd.SoftmaxActivation(nd.transpose(class_predictions, (0, 2, 1)), mode='channel')
            output = MultiBoxDetection(*[cls_probs, box_predictions, default_anchors], force_suppress=True, clip=False)
            output = box_nms(output, overlap_thresh=overlap_thres, valid_thresh=min_c, topk=topk)
            output = output.asnumpy()

            number_of_bbs = 0
            predicted_bb = []
            for b in range(output.shape[0]):
                predicted_bb_ = output[b, output[b, :, 0] != -1]
                predicted_bb_ = predicted_bb_[:, 2:]
                number_of_bbs += predicted_bb_.shape[0]
                predicted_bb_[:, 2] = predicted_bb_[:, 2] - predicted_bb_[:, 0]
                predicted_bb_[:, 3] = predicted_bb_[:, 3] - predicted_bb_[:, 1]
                predicted_bb.append(predicted_bb_)
            labels = y[:, :, 1:].asnumpy()
            labels[:, :, 2] = labels[:, :, 2] - labels[:, :, 0]
            labels[:, :, 3] = labels[:, :, 3] - labels[:, :, 1]

            with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
                output_image = draw_boxes_on_image(predicted_bb, labels, x.asnumpy())
                output_image[output_image<0] = 0
                output_image[output_image>1] = 1
                print("Number of predicted {} BBs = {}".format(print_name, number_of_bbs))
                sw.add_image('bb_{}_image'.format(print_name), output_image, global_step=e)

    total_loss = 0
    for loss in total_losses:
        total_loss = loss.asscalar()
    epoch_loss = float(total_loss)/len(dataloader)

    with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
        if update_metric:
            name1, val1 = cls_metric.get()
            name2, val2 = box_metric.get()
            sw.add_scalar(name1, {"test": val1}, global_step=e)
            sw.add_scalar(name2, {"test": val2}, global_step=e)
        sw.add_scalar('loss', {print_name: epoch_loss}, global_step=e)
            
    if save_cnn and e % save_every_n == 0 and e > 0:
        network.save_parameters("{}/{}".format(checkpoint_dir, checkpoint_name))
    return epoch_loss

# net.load_parameters("model_checkpoint/ssd_350.params")
for e in range(epochs):
    cls_metric.reset()
    box_metric.reset()

    log_dir = "./logs"
    train_loss = run_epoch(e, net, train_data, trainer, log_dir, print_name="train", update_cnn=True, update_metric=False, save_cnn=True)
    test_loss = run_epoch(e, net, test_data, trainer, log_dir, print_name="test", update_cnn=False, update_metric=True, save_cnn=False)
    if e % print_every_n == 0:
        name1, val1 = cls_metric.get()
        name2, val2 = box_metric.get()
        print("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}, test {3}={4:.6f}, {5}={6:.6f}".format(e, train_loss, test_loss,
                                                                                           name1, val1, name2, val2))
