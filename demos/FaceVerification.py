#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
import skimage

start = time.time()

import cv2
import sys
import numpy as np

np.set_printoptions(precision=2)
#sys.path.insert(1,'/home/vicident/Development/caffe_weiliu89/python')
sys.path.insert(1, '/home/vicident/Development/vk/caffe/python')
#MEAN_FILE_PATH = '/home/vicident/Development/vk/python/caffe/imagenet/ilsvrc_2012_mean.npy'

import caffe
import openface


class FaceVerification:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_representation(self, img_path):
        raise NotImplementedError

    @abstractmethod
    def cross_verification(self, img1, img2):
        raise NotImplementedError


def make_crops(orig, sizes, side):
    crops = []
    for size in sizes:
        sd = size - side
        resized = cv2.resize(orig, (size, size), interpolation=cv2.INTER_CUBIC)
        crops.append(resized[0:side, 0:side])
        crops.append(resized[0:side, -sd:0:-1])
        crops.append(resized[sd:size, 0:side])
        crops.append(resized[sd:size, -sd:0:-1])
        crops.append(resized[0:side, sd:size])
        crops.append(resized[0:side, -1:(sd-1):-1])
        crops.append(resized[sd:size, sd:size])
        crops.append(resized[sd:size, -1:(sd-1):-1])
        crops.append(resized[sd/2:(side+sd/2), sd/2:(side+sd/2)])
        crops.append(resized[sd/2:(side+sd/2), -sd/2:sd/2:-1])

    return crops


class OpenFaceVerification(FaceVerification):
    def __init__(self, dlibFacePredictor, network_model, image_dim=96, verbose=False):
        start = time.time()
        self.align = openface.AlignDlib(dlibFacePredictor)
        self.image_dim = image_dim
        self.net = openface.TorchNeuralNet(network_model, self.image_dim, cuda=True)
        self.verbose = verbose
        if verbose:
            print("Loading the dlib and OpenFace models took {} seconds.".format(time.time() - start))

    def get_representation(self, img_path, biggest_face_only=False):
        if self.verbose:
            print("Processing {}.".format(img_path))
        bgrImg = cv2.imread(img_path)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(img_path))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.verbose:
            print("  + Original size: {}".format(rgbImg.shape))

        start = time.time()
        if biggest_face_only:
            bbs = [self.align.getLargestFaceBoundingBox(rgbImg)]
        else:
            bbs = self.align.getAllFaceBoundingBoxes(rgbImg)
        if not len(bbs):
            raise Exception("Unable to find a face: {}".format(img_path))
        if self.verbose:
            print("  + Face detection took {} seconds.".format(time.time() - start))

        start = time.time()

        reps = []
        bbs_out = []
        for bb in bbs:
            if bb is not None:
                alignedFace = self.align.align(self.image_dim, rgbImg, bb,
                                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                if alignedFace is None:
                    print("Unable to align image: {}".format(img_path))
                    continue
                if self.verbose:
                    print("  + Face alignment took {} seconds.".format(time.time() - start))

                start = time.time()
                rep = self.net.forward(alignedFace)
                if self.verbose:
                    print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
                    print("Representation:")
                    print(rep)
                    print("-----\n")
                reps.append(rep)
                bbs_out.append(bb)
        return reps, bbs_out

    def cross_verification(self, img1, img2):
        l2_squared = []
        reps1, bbs1 = self.get_representation(img1)
        reps2, bbs2 = self.get_representation(img2)
        for i in xrange(len(reps1)):
            for j in xrange(len(reps2)):
                d = reps1[i] - reps2[j]
                l2_squared.append(np.dot(d, d))
                print("({}->{}: L2={:0.3f}".format(bbs1[i], bbs2[j], l2_squared[-1]))
        return bbs1, bbs2, reps1, reps2, l2_squared


class CaffeVerification(FaceVerification):
    def __init__(self, dlibFacePredictor, network_proto_path, network_model_path,
                 data_mean, layer_name, image_dim, verbose=False):
        start = time.time()
        self.align = openface.AlignDlib(dlibFacePredictor)
        self.image_dim = image_dim
        self.layer_name = layer_name
        self.net = caffe.Classifier(network_proto_path, network_model_path)
        self.mean = np.asarray([129.1863, 104.7624, 93.5940])
        self.verbose = verbose
        caffe.set_mode_gpu()
        self.counter = 0

        if verbose:
            print("Loading the dlib and Caffe models took {} seconds.".format(time.time() - start))

    def get_representation(self, img_path, biggest_face_only=False):
        if self.verbose:
            print("Processing {}.".format(img_path))
        bgrImg = cv2.imread(img_path)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(img_path))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.verbose:
            print("  + Original size: {}".format(rgbImg.shape))

        start = time.time()
        if biggest_face_only:
            bbs = [self.align.getLargestFaceBoundingBox(rgbImg)]
        else:
            bbs = self.align.getAllFaceBoundingBoxes(rgbImg)
        if not len(bbs):
            raise Exception("Unable to find a face: {}".format(img_path))
        if self.verbose:
            print("  + Face detection took {} seconds.".format(time.time() - start))

        start = time.time()

        reps = []
        bbs_out = []
        for bb in bbs:
            if bb is not None:
                #alignedFace = self.align.align(self.image_dim, rgbImg, bb,
                #                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                #if alignedFace is None:
                #    print("Unable to align image: {}".format(img_path))
                #    continue
                #if self.verbose:
                #    print("  + Face alignment took {} seconds.".format(time.time() - start))

                #alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2GRAY)
                #img = skimage.color.rgb2gray(alignedFace)
                #img = img[:, :, np.newaxis]
                h, w, c = rgbImg.shape
                img = rgbImg#[h/2-112:h/2+112, w/2-112:w/2+112]
                crops = make_crops(img, [256], 224)
                for i in xrange(len(crops)):
                    #cv2.imwrite("./temp/"+str(self.counter)+".jpg", crop)
                    #self.counter += 1
                    img = crops[i].astype(dtype=np.float)#alignedFace.astype(dtype=np.float)
                    img -= self.mean
                    img = img[:, :, (2, 1, 0)]
                    img = np.transpose(img, axes=[2, 0, 1])
                    self.net.blobs['data'].data[i, :, :, :] = img

                start = time.time()
                scores = self.net.forward()
                blobs = OrderedDict( [(k, v.data) for k, v in self.net.blobs.items()])
                rep = np.sum(blobs[self.layer_name]) / len(blobs[self.layer_name])
                if self.verbose:
                    print("  + Caffe forward pass took {} seconds.".format(time.time() - start))
                    print("Representation:")
                    print(rep)
                    print("-----\n")
                reps.append(rep)
                bbs_out.append(bb)
        return reps, bbs_out

    def cross_verification(self, img1, img2):
        l2_squared = []
        reps1, bbs1 = self.get_representation(img1)
        reps2, bbs2 = self.get_representation(img2)
        for i in xrange(len(reps1)):
            for j in xrange(len(reps2)):
                d = reps1[i] - reps2[j]
                l2_squared.append(np.dot(d, d))
                print("({}->{}: L2={:0.3f}".format(bbs1[i], bbs2[j], l2_squared[-1]))
        return bbs1, bbs2, reps1, reps2, l2_squared

if __name__ == '__main__':
    face_detector_path = '/home/vicident/Development/vk/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
    #network_proto_path = '/home/vicident/Development/AlfredXiangWu/proto/LightenedCNN_A_deploy.prototxt'
    #network_model_path = '/home/vicident/Development/AlfredXiangWu/model/LightenedCNN_A.caffemodel'
    network_proto_path = '/home/vicident/Development/vk/openface/models/vgg_face_caffe/VGG_FACE_deploy.prototxt'
    network_model_path = '/home/vicident/Development/vk/openface/models/vgg_face_caffe/VGG_FACE.caffemodel'
    data_mean = ''
    layer_name = 'fc6'
    image_dim = 224

    test_image_paths_1 = ['/home/vicident/Development/Bases/faces/lfw-deepfunneled/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0005.jpg',
                        '/home/vicident/Development/Bases/faces/lfw-deepfunneled/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0015.jpg',
                        '/home/vicident/Development/Bases/faces/lfw-deepfunneled/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0020.jpg']

    test_image_paths_2 = ['/home/vicident/Development/Bases/faces/lfw-deepfunneled/George_HW_Bush/George_HW_Bush_0003.jpg',
                          '/home/vicident/Development/Bases/faces/lfw-deepfunneled/George_HW_Bush/George_HW_Bush_0004.jpg',
                          '/home/vicident/Development/Bases/faces/lfw-deepfunneled/George_HW_Bush/George_HW_Bush_0012.jpg']

    verifier = CaffeVerification(face_detector_path,
                                 network_proto_path,
                                 network_model_path,
                                 data_mean,
                                 layer_name,
                                 image_dim)


    len1 = len(test_image_paths_1)
    len2 = len(test_image_paths_2)

    targets = []
    impostors = []

    for i in xrange(len1-1):
        for j in xrange(i+1, len1):
            bbs1, bbs2, reps1, reps2, l2_squared = verifier.cross_verification(test_image_paths_1[i], test_image_paths_1[j])
            targets += l2_squared

    for i in xrange(len2-1):
        for j in xrange(i+1, len2):
            bbs1, bbs2, reps1, reps2, l2_squared = verifier.cross_verification(test_image_paths_2[i], test_image_paths_2[j])
            targets += l2_squared

    for img1 in test_image_paths_1:
        for img2 in test_image_paths_2:
            bbs1, bbs2, reps1, reps2, l2_squared = verifier.cross_verification(img1, img2)
            impostors += l2_squared

    print "targets:", targets
    print "impostors:", impostors

    max_tar = max(targets)
    min_imp = min(impostors)

    print "max(targets) dist:", max_tar
    print "min(impostors) dist:", min_imp

    print "targets mean:", np.mean(targets)
    print "impostors mean:", np.mean(impostors)