#!/usr/bin/env python3

import sys

import numpy as np
import cv2
import h5py
from auxiliary.laserscan import LaserScan, SemLaserScan

import glob
import os
import yaml


if __name__ == '__main__':
    
    seq_id = sys.argv[1]

    laser_scan_object = LaserScan()
    CFG = yaml.load(open('config/semantic-kitti.yaml', 'r'))
    color_dict = CFG["color_map"]
    learning_map = CFG["learning_map"]
    print(learning_map)
    input()
    color_map_one_shot = {}
    for class_id, color in color_dict.items():
        if learning_map[class_id] in color_map_one_shot:
            continue
        else:
            color_map_one_shot[learning_map[class_id]] = color
    nclasses = len(color_dict)
    sem_laser_scan_object = SemLaserScan(nclasses, color_dict)
    #gt_labels = [filename for filename in sorted(glob.glob(os.path.join('/home/ayush/Downloads/sequence01/sequences/' + seq_id + '/labels/', '*.label')))]
    scans = [filename for filename in sorted(glob.glob(os.path.join('/home/ayush/Downloads/sequence01/sequences/' + seq_id + '/velodyne/', '*.bin')))]

    training_images = np.zeros((len(scans), 64, 1024, 5), np.float32)
    #gt_images = np.zeros((len(scans), 64, 1024), np.float32)
    #training_images = np.zeros((100, 64, 1024, 5), np.float32)
    #gt_images = np.zeros((100, 64, 1024), np.float32)

    training_data = zip(scans, gt_labels)
    counter = 0
    for scan_filename in scans:
        sem_laser_scan_object.open_scan(scan_filename)
        #sem_laser_scan_object.open_label(gt_filename)
        sem_laser_scan_object.do_range_projection()
        #sem_laser_scan_object.do_label_projection()

        range_image = sem_laser_scan_object.proj_range
        intensity = sem_laser_scan_object.proj_remission
        xyz = sem_laser_scan_object.proj_xyz

        x = xyz[:, :, 0]
        y = xyz[:, :, 1]
        z = xyz[:, :, 2]

        x = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=2)
        z = np.expand_dims(z, axis=2)

        depth = (range_image * 500)/65536
        depth = cv2.convertScaleAbs(depth, alpha=255)
        depth = np.float32(depth)
        depth = np.expand_dims(depth, axis=2)


        intensity = intensity + 1.
        intensity /= 2.0
        intensity = cv2.convertScaleAbs(intensity, alpha=255)
        intensity = np.float32(intensity)
        intensity = np.expand_dims(intensity, 2)

        training_image = depth
        training_image = np.concatenate((training_image, intensity), axis=2)
        training_image = np.concatenate((training_image, x), axis=2)
        training_image = np.concatenate((training_image, y), axis=2)
        training_image = np.concatenate((training_image, z), axis=2)
        # label = sem_laser_scan_object.proj_sem_label
        #


        training_images[counter, :, :, :] = training_image
        # gt_images[counter, :, :] = label
        # label = np.expand_dims(label, 2)
        # label_colorized = np.zeros((64, 1024, 3), np.uint8)
        # #print(np.min(label))
        # #print(np.max(label))
        #
        # for class_id, color in color_map_one_shot.items():
        #     mask = np.all(label == class_id, axis=-1)
        #     label_colorized[mask] = color
        # #cv2.namedWindow('image')
        # #cv2.imshow('image', np.float32(label == 1))
        # #cv2.waitKey()


        # label_colorized = cv2.cvtColor(label_colorized, cv2.COLOR_BGR2RGB)

        counter += 1
        #if counter > 99:
          #  break
    filename = '/home/dewan/data_training/dataset/sequences/' + seq_id + '/training_data.hdf5'
    hf = h5py.File(filename, 'w')
    hf.create_dataset('data', data=training_images)
    #hf.create_dataset('label', data=gt_images)
    hf.close()
       


