
# coding: utf-8

# In[ ]:

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf

from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
import json
import random
import operator, os
from functools import reduce
import tqdm

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))


# In[2]:


# some constants
NUM_CLASSES = 10
input_shape = (300, 300, 3)


# In[3]:


priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

# In[4]:


gt = pickle.load(open('gt_pascal.pkl', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)

labels = {0: [1, 0, 0, 0, 0, 0, 0, 0, 0], 1: [0, 0, 0, 0, 1, 0, 0, 0, 0], 2: [0, 1, 1, 1, 0, 0, 0, 0, 0],
          3: [0, 0, 1, 1, 0, 0, 0, 0, 0], 4: [1, 1, 1, 0, 0, 0, 0, 0, 0], 5: [1, 0, 1, 0, 0, 0, 0, 0, 0],
          6: [0, 0, 0, 1, 0, 0, 0, 0, 0], 7: [1, 1, 0, 0, 0, 0, 0, 0, 0], 8: [0, 0, 0, 0, 0, 1, 0, 0, 0],
          9: [0, 0, 0, 0, 0, 1, 1, 0, 0], 10: [0, 1, 0, 0, 1, 0, 0, 0, 0], 11: [0, 1, 1, 0, 1, 0, 0, 0, 0],
          12: [0, 0, 1, 0, 1, 0, 0, 0, 0], 13: [0, 0, 1, 0, 0, 1, 0, 0, 0], 14: [0, 1, 0, 0, 0, 1, 0, 0, 0],
          15: [0, 1, 0, 0, 1, 0, 1, 0, 0], 16: [0, 1, 1, 0, 1, 0, 1, 0, 0], 17: [0, 1, 1, 0, 0, 1, 1, 0, 0],
          18: [0, 0, 1, 0, 0, 1, 1, 0, 0], 19: [0, 1, 0, 1, 0, 0, 0, 0, 0], 20: [0, 0, 0, 1, 1, 0, 0, 0, 0],
          21: [0, 1, 0, 0, 0, 1, 0, 1, 0], 22: [0, 1, 1, 0, 0, 1, 0, 0, 0], 23: [0, 1, 0, 0, 0, 0, 0, 1, 0],
          24: [0, 0, 0, 0, 0, 0, 1, 0, 0], 25: [0, 1, 0, 1, 0, 0, 0, 0, 1], 26: [0, 0, 0, 1, 0, 0, 0, 0, 1],
          27: [0, 0, 0, 0, 1, 0, 0, 0, 1], 28: [0, 1, 0, 0, 1, 0, 0, 1, 0], 29: [0, 1, 1, 0, 1, 0, 0, 1, 0],
          30: [0, 0, 1, 0, 1, 0, 0, 1, 0], 31: [0, 1, 0, 0, 0, 1, 1, 0, 0], 32: [0, 0, 0, 0, 1, 0, 0, 1, 0],
          33: [0, 0, 0, 0, 0, 1, 0, 1, 0], 34: [0, 0, 0, 0, 0, 0, 0, 1, 0], 35: [0, 0, 0, 0, 0, 1, 0, 0, 1],
          36: [0, 1, 0, 0, 0, 0, 0, 0, 0], 37: [0, 0, 0, 0, 1, 0, 1, 0, 0], 38: [1, 0, 0, 0, 0, 0, 0, 0, 1],
          39: [1, 1, 0, 0, 0, 0, 0, 0, 1], 40: [0, 0, 0, 0, 1, 1, 0, 0, 0], 41: [0, 1, 0, 0, 1, 0, 0, 0, 1],
          42: [0, 1, 1, 0, 0, 1, 0, 0, 1], 43: [0, 1, 0, 0, 0, 0, 0, 1, 1]}


class JAAD_Generator(object):
    def __init__(self, bbox_util, batch_size, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.bbox_util = bbox_util
        self._imgpath = []
        self.batch_size = batch_size
        self.trainlist, self.testlist = self.make_lists()
        self.train_batches = len(self.trainlist) // self.batch_size
        self.val_batches = len(self.testlist) // self.batch_size
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def make_lists(self):
        all_folds = pickle.load(open('data/spilts.pickle', 'rb'))
        random.shuffle(all_folds)
        _test_folds = all_folds[-1]
        _train_folds = all_folds[:-1]
        train_folds = reduce(operator.add, _train_folds)
        test_folds = _test_folds

        return train_folds, test_folds

    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y + h, x:x + w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                    y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets

    def transform(self, width, height, annotation_path):
        anno = json.load(open(annotation_path))
        res = []
        for t in range(len(anno)):
            bbox = anno[t]
            label = bbox[0]
            '''pts = ['xmin', 'ymin', 'xmax', 'ymax']'''
            bndbox = []
            for i in range(4):
                scale = width if i % 2 == 0 else height
                cur_pt = min(scale, int(bbox[1][i]))
                cur_pt = float(cur_pt) / scale
                bndbox.append(cur_pt)
            bndbox.extend(labels[label])
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return np.asarray(res)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

    def generate(self, train=True):
        if train:
            keys = self.trainlist
        else:
            keys = self.testlist
        self._imgpath = \
            list(map(lambda x: x.replace("anno_full", "JAAD_frames").replace(".json", ".png"), keys))
        while True:

            inputs = []
            targets = []
            for j in range(len(keys)):
                img_path = self._imgpath[j]
                basename = os.path.basename(img_path).split(".")[0]
                dirname = os.path.dirname(img_path)
                img = imread(os.path.join(dirname, "frame_{:04d}.png".format(int(basename) + 1))).astype('float32')
                width, height, channel = img.shape
                y = self.transform(width, height, keys[j])
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype('float32')
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets

# In[6]:


path_prefix = '../../frames/'
# gen = JAAD_Generator(gt, bbox_util, 16, '../../frames/',
#                 train_keys, val_keys,
#                 (input_shape[0], input_shape[1]), do_crop=False)
gen = JAAD_Generator(bbox_util, 16, (input_shape[0], input_shape[1]))


# In[7]:


model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('checkpoints/weights.01-2.14.hdf5', by_name=True)


# In[8]:


freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

for L in model.layers:
    if L.name in freeze:
        L.trainable = False


# In[9]:


def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]


# In[10]:


base_lr = 1e-5
optim = keras.optimizers.sgd(lr=base_lr)
# optim = keras.optimizers.adam(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)


# In[11]:


num_epoch = 30
history = model.fit_generator(gen.generate(True), steps_per_epoch=gen.train_batches,
                              epochs=num_epoch,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              validation_steps=gen.val_batches,
                              workers=8)


# In[12]:


inputs = []
images = []
img_path = path_prefix + sorted(val_keys)[0]
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))


# In[13]:


preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)


# In[14]:


for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
#         label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

#     plt.show()

