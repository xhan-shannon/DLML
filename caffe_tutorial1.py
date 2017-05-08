# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
import sys

caffe_root = "/home/nco/DL/caffe/"
sys.path.insert(0, caffe_root+"python")

import caffe

caffe.set_mode_cpu()
caffe_def = caffe_root + "models/bvlc_reference_caffenet/deploy.prototxt"
caffe_weights = caffe_root + "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"

net = caffe.Net(caffe_def, caffe_weights, caffe.TEST)

mu = np.load(caffe_root + "python/caffe/imagenet/ilsvrc_2012_mean.npy")
mu = mu.mean(1).mean(1)
print "mean-subtracted values:", zip('BGR', mu)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

# set image size
net.blobs['data'].reshape(50, 3, 227, 227)

#image = caffe.io.load_image(caffe_root + "examples/images/cat.jpg")
image = caffe.io.load_image(caffe_root + "data/custom_images/test_pic5.jpg")
transformed_image = transformer.preprocess('data', image)

plt.imshow(image)
plt.show()

net.blobs['data'].data[...] = transformed_image
output = net.forward()
#t1 = Timer("net.forward()")
#print timeit("net.forward()")

output_prob = output['prob'][0]
print "predicted class is:", output_prob.argmax()

labels_file = caffe_root + "data/ilsvrc12/synset_words.txt"
labels = np.loadtxt(labels_file, str, delimiter="\t")
print "output label:", labels[output_prob.argmax()]

top_inds = output_prob.argsort()[::-1][:5]
print "probabilities and labels", zip(output_prob[top_inds], labels[top_inds])
