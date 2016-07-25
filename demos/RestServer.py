__author__ = 'vicident'

import os
from flask import Flask, jsonify, abort, request, make_response, url_for
from flask.ext.httpauth import HTTPBasicAuth
import urllib
import uuid
import copy
import argparse
from FaceVerification import OpenFaceVerification, CaffeVerification

# Local paths to bases and models
LOCAL_BASES_PATH = "/home/vicident/Development/Bases/faces"
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--interface', type=str, help="IP to bind", default="192.168.1.101")
args = parser.parse_args()

# Rest application
app = Flask(__name__, static_url_path="")
auth = HTTPBasicAuth()

#verificator = OpenFaceVerification(args.dlibFacePredictor, args.networkModel, args.imgDim, args.verbose)
face_detector_path = '/home/vicident/Development/vk/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
#network_proto_path = '/home/vicident/Development/AlfredXiangWu/proto/LightenedCNN_A_deploy.prototxt'
#network_model_path = '/home/vicident/Development/AlfredXiangWu/model/LightenedCNN_A.caffemodel'
network_proto_path = '/home/vicident/Development/vk/openface/models/vgg_face_caffe/VGG_FACE_deploy.prototxt'
network_model_path = '/home/vicident/Development/vk/openface/models/vgg_face_caffe/VGG_FACE.caffemodel'
data_mean = ''
layer_name = 'fc6'
image_dim = 224

verificator = CaffeVerification(face_detector_path,
                             network_proto_path,
                             network_model_path,
                             data_mean,
                             layer_name,
                             image_dim,
                             args.verbose)

response_template_ver = {
    "etalon":
    {
        '#faces': 0,
        'bboxes': []
    },
    "test":
    {
        '#faces': 0,
        'bboxes': []
    },
    "ver":
    {
        'scores': []
    }
}

response_template_emb = {
    "image":
    {
        '#faces': 0,
        'bboxes': [],
        'embeddings': []
    }
}


def convert_dlib_rect_list_to_str(rects):
    string = ""
    for i in xrange(len(rects) - 1):
        string += "{0}, {1}, {2}, {3};\n".format(rects[i].left(), rects[i].top(), rects[i].right(), rects[i].bottom())
    string += "{0}, {1}, {2}, {3}".format(rects[-1].left(), rects[-1].top(), rects[-1].right(), rects[-1].bottom())
    return string


def convert_dlib_rect_to_list(rects):
    rect_list = []
    for rect in rects:
        rect_list.append([rect.left(), rect.top(), rect.right(), rect.bottom()])
    return rect_list


def convert_list_to_str(list):
    string = ""
    for i in xrange(len(list) - 1):
        string += str(list[i]) + "; "
    string += str(list[-1])
    return string


@auth.get_password
def get_password(username):
    if username == 'vk':
        return 'vk'
    elif username == 'admin':
        return 'admin'
    return None


@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 403)
    # return 403 instead of 401 to prevent browsers from displaying the default auth dialog


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/openface/ver/', methods=['GET'])
@auth.login_required
def verify_image_pair():
    get_args = request.args
    print get_args
    print len(get_args)
    if 'etalon' in get_args.keys() and 'test' in get_args.keys():
        etalon, test = get_args['etalon'], get_args['test']
        imagefile = urllib.URLopener()
        print "etalon:", etalon
        print "test:", test
        etalon_name = "/tmp/etalon_" + str(uuid.uuid1()) + ".jpg"
        imagefile.retrieve(etalon, etalon_name)
        test_name = "/tmp/test_" + str(uuid.uuid1()) + ".jpg"
        imagefile.retrieve(test, test_name)
        bbs1, bbs2, _, _, l2_squared = verificator.cross_verification(etalon_name, test_name)
        response = cver_results_to_json(bbs1, bbs2, l2_squared)
        return make_response(jsonify({'answer': response}))
    else:
        print "parameters are not defined"
        return make_response(jsonify({'error': 'Please, define image pair: ?etalon=<URL>&test=<URL>'}), 400)


@app.route('/openface/emb/', methods=['GET'])
@auth.login_required
def make_embeddings():
    try:
        get_args = request.args
        print get_args
        biggest_face_only = False
        if 'image' in get_args.keys():
            image_url = get_args['image']
            if 'bfonly' in get_args.keys():
                biggest_face_only = int(get_args['bfonly'])
            if "://" in image_url:
                image_file = urllib.URLopener()
                print "image_url:", image_url
                image_name = "/tmp/image_" + str(uuid.uuid1()) + ".jpg"
                image_file.retrieve(image_url, image_name)
            else:
                image_url = image_url.replace("\n", "")
                image_name = LOCAL_BASES_PATH + os.path.sep + image_url
            reps, bbs = verificator.get_representation(image_name, biggest_face_only)
            response = emb_results_to_json(bbs, reps)
            return make_response(jsonify({'answer': response}))
        else:
            print "parameters are not defined"
            return make_response(jsonify({'error': 'Please, define image URL: ?image=<URL>'}), 400)
    except:
        return make_response(jsonify({'error': 'Exception has been raised during the session. Please try again'}), 400)


def cver_results_to_json(bbs1, bbs2, l2_squared):
    response = copy.deepcopy(response_template_ver)
    response['etalon']['#faces'] = len(bbs1)
    response['etalon']['bboxes'] = convert_dlib_rect_to_list(bbs1)
    response['test']['#faces'] = len(bbs2)
    response['test']['bboxes'] = convert_dlib_rect_to_list(bbs2)
    response['ver']['scores'] = l2_squared

    return response


def emb_results_to_json(bbs, reps):
    response = copy.deepcopy(response_template_emb)
    response['image']['#faces'] = len(bbs)
    response['image']['bboxes'] = convert_dlib_rect_to_list(bbs)
    reps_list = [rep.tolist() for rep in reps]
    response['image']['embeddings'] = reps_list
    return response


if __name__ == '__main__':
    app.run(debug=True, host=args.interface)
