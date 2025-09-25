import numpy as np
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import Tensor, Parameter
import mindspore.nn.probability.distribution as msd

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed


def softmax_cross_entropy(logits, labels, smooth_factor=0.0):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    num_class = logits.shape[-1]
    loss_xe = -mnp.sum(labels * nn.LogSoftmax(-1)(logits), axis=-1)
    loss_smooth = -mnp.sum( (mnp.ones_like(labels)/num_class) * nn.LogSoftmax(-1)(logits), axis=-1)
    loss = (1.-smooth_factor)*loss_xe + smooth_factor*loss_smooth
    return mnp.asarray(loss)


def sigmoid_cross_entropy(logits, labels):
    """Computes sigmoid cross entropy given logits and multiple class labels."""
    log_p = nn.LogSigmoid()(logits)
    log_not_p = nn.LogSigmoid()(-logits)
    loss = -labels * log_p - (1. - labels) * log_not_p
    return mnp.asarray(loss)


class OrdinalXCE(nn.Cell):
    """Return regularized loss for multi-class classifications of Ordinal variables.
    
    """
    def __init__(self, num_class, e=0.0, neighbors=1):
        super(OrdinalXCE, self).__init__()
        
        self.num_class = num_class
        self.e = e # label_smoothing
        self.neighbors = neighbors

        ### self.neighbors -> self.label_smoothing
        neighbor_mask = np.ones((self.num_class,self.num_class))
        neighbor_mask = neighbor_mask - np.triu(neighbor_mask, neighbors) - np.tril(neighbor_mask, -neighbors)
        neighbor_mask = neighbor_mask / (np.sum(neighbor_mask,axis=-1,keepdims=True) + 1e-6)
        self.neighbor_mask = Tensor(neighbor_mask, mnp.float32)
        
        self.cross_entropy = softmax_cross_entropy

    def construct(self, prediction_logits, target_tensor):
        '''
        prediction_logits is the output tensor (without softmax) with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor, one-hot codes.
        '''
        
        # (None,bins):
        xent_loss = self.cross_entropy(logits=prediction_logits, labels=target_tensor)
        
        ### add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        smoothed_labels = P.MatMul()(target_tensor, self.neighbor_mask)
        # (None,):
        smoothed_xent = self.cross_entropy(logits=prediction_logits, labels=smoothed_labels)
        
        final_loss = (1-self.e)*xent_loss + self.e*smoothed_xent
        return final_loss


### @ZhangJ. 修正了MultiClassFocalLoss中关于LabelSmoothing的bugs.
class MultiClassFocalLoss(nn.Cell):
    """Return Focal loss for multi-class classifications.
    
    """
    def __init__(self, num_class=10, beta=0.99, gamma=2., e=0.1, not_focal=False, balanced=False):
        super(MultiClassFocalLoss, self).__init__()
        self.num_class = num_class
        self.beta = beta
        self.gamma = gamma
        self.e = e ### Label Smoothing Factor
        self.not_focal = not_focal
        self.balanced = balanced

        ### self.neighbors -> self.label_smoothing
        neighbor_mask = np.ones((self.num_class,self.num_class))
        self.neighbor_mask = Tensor(neighbor_mask, mstype.float32)

        self.class_weights = ms.Parameter(Tensor(np.ones((self.num_class))/self.num_class, dtype=mstype.float32), \
            name='class_weights', requires_grad=False)

        self.softmax = nn.Softmax(-1)
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.zero = ms.Tensor([0.])

        # self.onehot = nn.OneHot(depth=self.num_bins)
        self.allreduce = P.Identity()
        self.device_num = 1
        if distributed:
            self.allreduce = P.AllReduce()
            self.device_num = D.get_group_size()
    
    def _compute_classes_num(self,target_tensor):
        # (100,):
        classes_num = mnp.sum(target_tensor, 0)
        classes_num = self.allreduce(classes_num)
        classes_num = F.cast(classes_num, mstype.float32)
        #@ZhangJ. Added this for robustness:
        classes_num += 1.
        return classes_num

    def construct(self, prediction_logits, target_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor, one-hot codes.
        '''
        prediction_logits = F.cast(prediction_logits, mnp.float32)
        target_tensor = F.cast(target_tensor, mnp.float32)
        
        prediction_tensor = self.softmax(prediction_logits)

        #1# get focal loss with no balanced weight which presented in paper function (4)
        # (none,100):
        zeros = mnp.zeros_like(prediction_tensor)
        one_minus_p = mnp.where(target_tensor>1e-5, target_tensor-prediction_tensor, zeros)
        # (None,100):
        FT = -1 * mnp.power(one_minus_p,self.gamma) * mnp.log(mnp.clip(prediction_tensor, 1e-8, 1.0))
        
        alpha = mnp.ones_like(FT)
        
        if self.balanced:
            #2# get balanced weight alpha
            # (100,):
            classes_num = self._compute_classes_num(target_tensor)
            total_num = mnp.sum(classes_num)

            # (100,):
            classes_w_t1 = total_num/classes_num
            sum_ = mnp.sum(classes_w_t1)
            # (100,):
            classes_w_t2 = classes_w_t1/sum_ # normalized scale.
            classes_w_tensor = F.cast(classes_w_t2, mstype.float32)
            ## classes_w_tensor sums to 1.

            ### Perform EMA over weights:
            # (100):
            weights = self.beta*self.class_weights + (1-self.beta)*classes_w_tensor
            ## weights sums to 1.
            P.Assign()(self.class_weights, weights)

            # (None,100):
            classes_weight = mnp.broadcast_to(mnp.expand_dims(weights,0), target_tensor.shape)
            # (None,100):
            alpha = mnp.where(target_tensor>zeros, classes_weight, zeros)

        ### get balanced focal loss
        # (None,100):
        balanced_fl = alpha * FT
        # (None,):
        balanced_fl = mnp.sum(balanced_fl,-1)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        smoothed_labels = mnp.ones_like(target_tensor) / self.num_class
        # 用于label_smoothing的loss, (None,), (None,100):
        xent, _dlogits = self.cross_entropy(prediction_logits, smoothed_labels)
        
        '''
        #4# add other op to prevent overfit # https://github.com/maozezhong/focal_loss_multi_class/blob/master/focal_loss.py
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)
        '''
        
        # (None,):
        final_loss = (1-self.e)*balanced_fl + self.e*xent
        
        if self.not_focal:
            labels = target_tensor
            # (None,), (None,100):
            softmax_xent, _dlogits = self.cross_entropy(prediction_logits, labels)
            final_loss = (1-self.e)*softmax_xent + self.e*xent
            
        return final_loss


class BinaryFocalLoss(nn.Cell):
    """Return Focal loss for Binary classifications.
    
    """
    def __init__(self, alpha=0.25, gamma=2., feed_in_logit=False, not_focal=False): ### Pass config.model.heads.X here
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.feed_in_logit = feed_in_logit
        self.not_focal = not_focal

        self.cross_entropy = P.BinaryCrossEntropy(reduction='none')
        self.sigmoid = P.Sigmoid()
        
    def _convert_logit(self, probs):
        probs = mnp.clip(probs,1e-5,1.-1e-5)
        logits = mnp.log(probs/(1-probs))
        return logits

    def construct(self, logits, labels):
        '''
        logits is the output tensor with shape [None,] before sigmoid;
        labels is the label tensor, [None,], 0 or 1 valued.
        1: dist<15; 0: dist>15. (Compute the distribution of these two labels)
        '''
        epsilon = 1e-8
        
        labels = F.cast(labels, mstype.float32)
        probs = F.cast(logits, mstype.float32)
        if self.feed_in_logit:
            probs = self.sigmoid(logits)
        else:
            logits = self._convert_logit(logits)
        
        # (None):
        _ones = mnp.ones_like(labels)
        positive_pt = mnp.where(labels>1e-5, probs, _ones)
        negative_pt = mnp.where(labels<1e-5, 1-probs, _ones)
        
        # (None,):
        focal_loss = -self.alpha * mnp.power(1-positive_pt, self.gamma) * mnp.log(mnp.clip(positive_pt, epsilon, 1.)) - \
            (1-self.alpha) * mnp.power(1-negative_pt, self.gamma) * mnp.log(mnp.clip(negative_pt, epsilon, 1.))
        focal_loss *= 2.
        
        if self.not_focal:
            focal_loss = self.cross_entropy(logits, labels, _ones)
        
        return focal_loss
    

class ArgMax_Loss(nn.Cell):
    """Define ArgMax-Loss for CARL Loss.
    CARL: Classification-Augmented Regression-Like Losses
    """
    def __init__(self, config):
        super(ArgMax_Loss, self).__init__()
        # config = config...estogram
        
        self.charbonnier_eps = config.charbonnier_eps
        self.softmax_temperature = config.softmax_temperature
        self.num_sample = config.num_sample
        self.gaussian_width_factor = config.gaussian_width_factor ### = 2 or 3

        self.first_break = config.first_break
        self.last_break = config.last_break
        self.num_bins = config.num_bins
        
        self.breaks = mnp.linspace(self.first_break, self.last_break, self.num_bins)
        self.width = self.breaks[1] - self.breaks[0]
        # ->(Nbins):
        self.centers = self.breaks - 0.5*self.width
        
        self.lower_bound = self.breaks[0]
        self.upper_bound = self.breaks[-2] # @ZhangJ. -2 instead of -1
        
        self.std = self.width/self.gaussian_width_factor
        self.clamp_min = self.width / 2.

        self.softmax = nn.Softmax(-1)
        self.uniform = msd.Uniform(ms.Tensor(1e-5,dtype=mnp.float32), ms.Tensor(1.-1e-5,dtype=mnp.float32), dtype=mnp.float32)
        self.normal = msd.Normal(ms.Tensor(0.,dtype=mnp.float32), ms.Tensor(1,dtype=mnp.float32), dtype=mnp.float32)
        self.zero = ms.Tensor([0.])
    
    def gumbel_softmax(self, logits):
        gumbel_eps = self.uniform.sample(logits.shape)
        gumbel_eps = F.cast(gumbel_eps, logits.dtype)
        # (...,bins) same as logits.shape:
        g_softmax = self.softmax((logits - mnp.log(-mnp.log(gumbel_eps))/self.softmax_temperature))
        return g_softmax

    def sampling_argmax(self, softmax_prob):
        # softmax_prob: (...,bins)
        nres = softmax_prob.shape[0]

        # (1,...,bins):
        prob = mnp.reshape(softmax_prob, (1,-1,self.num_bins))
        # (B,...,bins):
        prob = mnp.tile(prob, (self.num_sample,1,1))
        
        gaussian_eps = self.normal.sample(shape=prob.shape)
        # (B,...,bins):
        gaussian_samples = mnp.reshape(self.centers, (1,1,-1)) + self.std*gaussian_eps
        gaussian_samples = mnp.clip(gaussian_samples, self.lower_bound, self.upper_bound)

        # (B,...):
        drawn_samples = mnp.sum(prob*gaussian_samples, axis=-1) ### Samples from Gaussian Mixture Model
        
        return drawn_samples

    def charbonnier_loss_fn(self, x, y):
        ### Compute Absolute Errors:
        charbonnier_error = mnp.sqrt(mnp.square(x-y) + self.charbonnier_eps)
        error = mnp.clip(charbonnier_error, self.clamp_min, 100.0) - self.clamp_min
        return error

    ### @ZhangJ. 先采样y_pred, 再算误差, 再算期望
    def construct(self, logits, labels):
        ### logits: (...,bins); labels: (...)
        ### logits: (Nab,bins)
        
        # (...,bins):
        categorical_prob = self.gumbel_softmax(logits)
        # (B,...):
        drawn_samples = self.sampling_argmax(categorical_prob)

        ### Compute error per sample:
        # (1,...):
        y = mnp.expand_dims(labels, axis=0)
        # (B,Nab=...):
        errors = self.charbonnier_loss_fn(drawn_samples, y)
        # (Nab):
        errors = mnp.mean(errors, axis=0)

        return errors, drawn_samples
    
    # def construct(self, logits, labels):
    #     ### logits: (...,bins); labels: (...)
    #     ### logits: (Nab,bins)

    #     # (Nab,bins):
    #     estogram = self.charbonnier_loss_fn(mnp.reshape(self.centers, (1,-1)), mnp.reshape(labels, (-1,1)))

    #     # (...,bins):
    #     categorical_prob = self.gumbel_softmax(logits)
    #     # (B,...):
    #     drawn_samples = self.sampling_argmax(categorical_prob)

    #     ### @@@@@@@@@@@
        
    #     # (...,bins):
    #     categorical_prob = self.gumbel_softmax(logits)
    #     # (B,...):
    #     drawn_samples = self.sampling_argmax(categorical_prob)

    #     ### Compute error per sample:
    #     # (1,...):
    #     y = mnp.expand_dims(labels, axis=0)
    #     # (B,Nab=...):
    #     errors = self.charbonnier_loss_fn(drawn_samples, y)
    #     # (Nab):
    #     errors = mnp.mean(errors, axis=0)

    #     return errors, drawn_samples
    