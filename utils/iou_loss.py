from mxnet.gluon.loss import Loss
from mxnet import gluon

class IOU_loss(Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(IOU_loss, self).__init__(weight, batch_axis, **kwargs)
        
    def hybrid_forward(self, F, pred, label, sample_weight=None):
        # source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # Yu, J., Jiang, Y., Wang, Z., Cao, Z., & Huang, T. (2016, October). Unitbox: An advanced object detection network.
        # In Proceedings of the 2016 ACM on Multimedia Conference (pp. 516-520). ACM.
        pred_area = pred[:, 2] * pred[:, 3]
        label_area = label[:, 2] * label[:, 3]

        x1_1, y1_1 = pred[:, 0], pred[:, 1]
        x2_1, y2_1 = pred[:, 0] + pred[:, 2], pred[:, 1] + pred[:, 3]

        x1_2, y1_2 = label[:, 0], label[:, 1]
        x2_2, y2_2 = label[:, 0] + label[:, 2], label[:, 1] + label[:, 3]

        x_overlaps = F.logical_or(x2_1 < x1_2, x1_1 > x2_2)
        y_overlaps = F.logical_or(y2_1 < y1_2, y1_1 > y2_2)
        overlaps = F.logical_not(F.logical_or(x_overlaps, y_overlaps))

        x1_1 = x1_1.expand_dims(0)
        y1_1 = y1_1.expand_dims(0)
        x2_1 = x2_1.expand_dims(0)
        y2_1 = y2_1.expand_dims(0)
        x1_2 = x1_2.expand_dims(0)
        y1_2 = y1_2.expand_dims(0)
        x2_2 = x2_2.expand_dims(0)
        y2_2 = y2_2.expand_dims(0)
        
        x_a = F.max(F.concat(x1_1, x1_2, dim=0), axis=0)
        y_a = F.max(F.concat(y1_1, y1_2, dim=0), axis=0)
        x_b = F.min(F.concat(x2_1, x2_2, dim=0), axis=0)
        y_b = F.min(F.concat(y2_1, y2_2, dim=0), axis=0)
        
        inter_area = (x_b - x_a) * (y_b - y_a)

        iou = F.log(inter_area) - F.log(pred_area + label_area - inter_area)
        #iou = inter_area / (pred_area + label_area - inter_area)
        
        loss = gluon.loss._apply_weighting(F, iou, self._weight, sample_weight)
        loss = F.where(F.logical_not(overlaps), F.zeros(shape=overlaps.shape), loss)
        mean_loss = F.mean(loss, axis=self._batch_axis, exclude=True)
        #print("ml = {}".format(mean_loss))
        return -mean_loss
