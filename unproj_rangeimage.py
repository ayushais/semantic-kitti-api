import numpy as np
import cv2

import sys
import numpy as np
import cv2
import h5py
from auxiliary.laserscan import LaserScan, SemLaserScan

import glob
import os
import yaml

import  h5py
import cv2
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':
    seq_id = sys.argv[1]

    laser_scan_object = LaserScan()
    CFG = yaml.load(open('config/semantic-kitti.yaml', 'r'))
    color_dict = CFG["color_map"]
    learning_map = CFG["learning_map"]
    color_map_one_shot = {}
    for class_id, color in color_dict.items():
        if learning_map[class_id] in color_map_one_shot:
            continue
        else:
            color_map_one_shot[learning_map[class_id]] = color
    nclasses = len(color_dict)
    sem_laser_scan_object = SemLaserScan(nclasses, color_dict)
    scans = [filename for filename in sorted(glob.glob(os.path.join('/home/ayush/Downloads/sequence01/sequences/' + seq_id + '/velodyne/', '*.bin')))]
    #gt_labels = [filename for filename in sorted(glob.glob(os.path.join('/home/ayush/Downloads/sequence01/sequences/' + seq_id + '/labels/', '*.label')))]
    prediction_labels = [filename for filename in sorted(glob.glob(os.path.join('/home/ayush/Downloads/sequence01/sequences/' + seq_id + '/predictions/', '*.png')))]
    count = 0
    training_data = dict(zip(scans, prediction_labels))
    for scan_filename, prediction_filename in training_data.items():
        sem_laser_scan_object.open_scan(scan_filename)
#     sem_laser_scan_object.open_label(gt_filename)
        sem_laser_scan_object.do_range_projection()
        range_image = sem_laser_scan_object.proj_range
        depth = (range_image * 500)/65536
        depth = cv2.convertScaleAbs(depth, alpha=255)

        prediction = cv2.imread(prediction_filename, -1)
#
    # for prediction_filename in prediction_labels:
        combined = np.zeros((128, 1024), np.uint8)
        combined[0:64, :] = depth
        combined[64:128, :] = (prediction == 1) * 255

        prediction = cv2.imread(prediction_filename)
        cv2.namedWindow("image")
        cv2.imshow("image", combined)
        cv2.waitKey(30)


    # training_data = dict(zip(scans, gt_labels))
    # for scan_filename, gt_filename in training_data.items():
    #     sem_laser_scan_object.open_scan(scan_filename)
    #     sem_laser_scan_object.open_label(gt_filename)
    #     sem_laser_scan_object.do_range_projection()
    #
    #     sem_laser_scan_object.do_label_projection()
    #
    #     label = sem_laser_scan_object.proj_sem_label
    #     proj_x = sem_laser_scan_object.proj_x
    #     proj_y = sem_laser_scan_object.proj_y
    #
    #     test_label = np.zeros((64, 1024), np.int32)
    #     test_label = np.zeros(proj_x.shape[0], np.int32)
    #     for index in range(proj_x.shape[0]):
    #         test_label[index] = train_label[count, proj_y[index], proj_x[index]]
    #         #test_label[proj_y[index], proj_x[index]] = train_label[count, proj_y[index], proj_x[index]]
    #     count += 1
    #
    #     #test_label = np.reshape(test_label, (-1))
    #     #print(np.sum(label - test_label))
    #     path = gt_filename.replace('labels', 'predictions')
    #
    #     test_label.tofile(path)
    #
    #     sem_laser_scan_object.open_label(path)
    #     sem_laser_scan_object.do_label_projection()
    #
    #     prediction = sem_laser_scan_object.proj_sem_label
    #
    #     print(np.sum(prediction - label))
    #
    #
    #     cv2.namedWindow("image")
    #     cv2.imshow("image", np.float32(prediction == 1))
    #     cv2.waitKey(30)
    #

