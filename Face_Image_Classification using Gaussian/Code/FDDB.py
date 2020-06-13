import os
import sys
import numpy as np
import random
import cv2
import math
import argparse
import pickle

def check_path(path):
    if not os.path.exists(path):
        print('Not found {}'.format(path))
        sys.exit(0)


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ellipse_to_bbox(obj):
    maj_rad = obj[0]
    min_rad = obj[1]
    angle = obj[2]
    xcenter = obj[3]
    ycenter = obj[4]

    cosin = math.cos(math.radians(-angle))
    sin = math.sin(math.radians(-angle))

    x1 = cosin * (-min_rad) - sin * (-maj_rad) + xcenter
    y1 = sin * (-min_rad) + cosin * (-maj_rad) + ycenter
    x2 = cosin * (min_rad) - sin * (-maj_rad) + xcenter
    y2 = sin * (min_rad) + cosin * (-maj_rad) + ycenter
    x3 = cosin * (min_rad) - sin * (maj_rad) + xcenter
    y3 = sin * (min_rad) + cosin * (maj_rad) + ycenter
    x4 = cosin * (-min_rad) - sin * (maj_rad) + xcenter
    y4 = sin * (-min_rad) + cosin * (maj_rad) + ycenter
    wid = [x1, x2, x3, x4]
    hei = [y1, y2, y3, y4]
    xmin_ = int(min(wid))
    xmax_ = int(max(wid))
    ymin_ = int(min(hei))
    ymax_ = int(max(hei))

    return xmin_, ymin_, xmax_, ymax_

def load_FDDB(image_dir, anno_dir, which_folds):
    """ origional annotation format
            2002/08/11/big/img_591
            1
            123.583300 85.549500 1.265839 269.693400 161.781200  1
            2002/08/26/big/img_265
            3
            67.363819 44.511485 -1.476417 105.249970 87.209036  1
            41.936870 27.064477 1.471906 184.070915 129.345601  1
            70.993052 43.355200 1.370217 340.894300 117.498951  1
        """
    # check directories of the dataset
    check_path(image_dir)
    check_path(anno_dir)
    dataset = []

    for fold in which_folds:
        anno_file = os.path.join(anno_dir, 'FDDB-fold-{:02d}-ellipseList.txt'.format(fold))
        check_path(anno_file)

        # load annotation file
        with open(anno_file, 'r') as f:
            lines = [i.replace('\n', '') for i in f.readlines()]
        curr_ind = 0
        while curr_ind < len(lines):
            sample = dict()
            sample['img_path'] = os.path.join(image_dir, lines[curr_ind] + '.jpg')
            curr_ind += 1

            num_faces = int(lines[curr_ind])
            sample['num_faces'] = num_faces
            curr_ind += 1

            bboxes = []
            for j in range(num_faces):
                ell = lines[curr_ind].split(' ')
                curr_ind += 1
                ell = [float(ell[i]) for i in range(5)]
                bbox = ellipse_to_bbox(ell)
                bboxes.append(bbox)

            sample['bboxes'] = np.array(bboxes)

            dataset.append(sample)

    return dataset


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    x2 = min(a[2], b[2])
    if x1 >= x2:
        return 0.0

    y1 = max(a[1], b[1])
    y2 = min(a[3], b[3])
    if y1 >= y2:
        return 0.0

    inter = (x2-x1) * (y2-y1)
    union = (max(a[2], b[2]) - min(a[0], b[0])) * (max(a[3], b[3]) - min(a[1], b[1]))

    return float(inter) / float(union)


def is_overlap(a, bboxes, threshold=0.3):
    for i in range(len(bboxes)):
        b = bboxes[i]
        ov = compute_iou(a, b)
        if ov >= threshold:
            return True

    return False


def create_datasets_FDDB(image_dir, anno_dir, which_folds,
                         save_dir, isTraining, useColor, dSize,
                         numPos=1000, numNeg=1000, doShuffle=False, negMaxOverlap=0.3, ignore_existed=True):

    # create directories for saving cropped datasets if necessary
    make_dir_if_not_exist(save_dir)
    dataset_tag = '{}-{}-{}'.format(dSize, 'color' if useColor else 'gray', 'Train' if isTraining else 'Test')
    dataset_root = os.path.join(save_dir, dataset_tag)
    make_dir_if_not_exist(dataset_root)

    pos_file = os.path.join(dataset_root, 'pos.bin')
    pos_dir = os.path.join(dataset_root, 'Pos')
    make_dir_if_not_exist(pos_dir)

    neg_file = os.path.join(dataset_root, 'neg.bin')
    neg_dir = os.path.join(dataset_root, 'Neg')
    make_dir_if_not_exist(neg_dir)

    # if existed, load them
    if not ignore_existed:
        if os.path.exists(pos_file) and os.path.exists(neg_file):
            with open(pos_file, 'rb') as f:
                pos_data = pickle.load(f)
            with open(neg_file, 'rb') as f:
                neg_data = pickle.load(f)

            return pos_data, neg_data

    # load annotation file
    tmp_folds = which_folds.split(',')
    folds = [int(i) for i in tmp_folds ]
    FDDB = load_FDDB(image_dir, anno_dir, folds)

    # shuffle or not
    num_sample = len(FDDB)
    idx_sample = range(num_sample)
    if doShuffle:
        random.shuffle(idx_sample)

    img_tag = cv2.IMREAD_COLOR if useColor else cv2.IMREAD_GRAYSCALE

    # start cropping
    num_pixels = dSize * dSize * (3 if useColor else 1)
    pos_data = np.empty([0, num_pixels], dtype = np.float32)
    neg_data = np.empty([0, num_pixels], dtype = np.float32)

    # crop pos
    numPos_Crop = 0
    for i in idx_sample:
        if numPos_Crop >= numPos:
            break

        sample = FDDB[i]
        img = cv2.imread(sample['img_path'], img_tag)  # BGR if color in opencv

        for j in range(sample['num_faces']):
            bbox = sample['bboxes'][j, :]
            if (bbox[0] > 0 and bbox[1] > 0 and bbox[2] < img.shape[1] and bbox[3] < img.shape[0]):
                if useColor:
                    face = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                else:
                    face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                face_resize = cv2.resize(src=face, dsize=(dSize, dSize), dst=None, interpolation=cv2.INTER_LINEAR)
                save_name = os.path.join(pos_dir, '{:06d}.jpg'.format(numPos_Crop))
                cv2.imwrite(save_name, face_resize)
                numPos_Crop += 1

                face_flatten = np.array(face_resize).flatten()
                pos_data = np.vstack((pos_data, face_flatten))

    # crop neg
    numNeg_Crop = 0
    for i in idx_sample:
        if numNeg_Crop >= numNeg:
            break

        sample = FDDB[i]
        img = cv2.imread(sample['img_path'], img_tag)  # BGR if color in opencv

        face_bboxes = []
        for j in range(sample['num_faces']):
            bbox = sample['bboxes'][j, :]
            face_bboxes.append(bbox)

        # crop neg
        num_try = 100
        neg_bboxes = []
        for j in range(sample['num_faces']):
            if numNeg_Crop >= numNeg:
                break

            num_tried = 0
            while True:
                if num_tried > num_try:
                    break

                x1 = random.randint(0, img.shape[1] - dSize)
                x2 = random.randint(x1+dSize, img.shape[1])
                y1 = random.randint(0, img.shape[0] - dSize)
                y2 = random.randint(y1+dSize, img.shape[0])
                bbox = [x1, y1, x2, y2]
                num_tried += 1

                if not is_overlap(bbox, face_bboxes, negMaxOverlap) and not is_overlap(bbox, neg_bboxes, negMaxOverlap):
                    if useColor:
                        neg = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                    else:
                        neg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    neg_resize = cv2.resize(src=neg, dsize=(dSize, dSize), dst=None, interpolation=cv2.INTER_LINEAR)
                    save_name = os.path.join(neg_dir, '{:06d}.jpg'.format(numNeg_Crop))
                    cv2.imwrite(save_name, neg_resize)
                    numNeg_Crop += 1
                    neg_bboxes.append(bbox)

                    neg_flatten = np.array(neg_resize).flatten()
                    neg_data = np.vstack((neg_data, neg_flatten))

                    break

    # save results
    with open(pos_file, 'wb') as f:
        pickle.dump(pos_data, f, pickle.HIGHEST_PROTOCOL)

    with open(neg_file, 'wb') as f:
        pickle.dump(neg_data, f, pickle.HIGHEST_PROTOCOL)

    return pos_data, neg_data


def parse_args():
    parser = argparse.ArgumentParser(description='Crop Face and Non-Face Patcehs from FDDB')
    # general
    parser.add_argument('--image_dir', help='path storing originalPics of FDDB', required=True, type=str)
    parser.add_argument('--annotation_dir', help='path storing annotation files of FDBB', required=True, type=str)
    parser.add_argument('--which_folds', help='which folds to use, e.g., 1,2,3', required=True, type=str)
    parser.add_argument('--save_dir', help='path storing crop images', required=True, type=str)
    parser.add_argument('--is_training', help='data for training or testing', default=True, type=bool)
    parser.add_argument('--use_color', help='use color image or not', default=True, type=bool)
    parser.add_argument('--patch_size', help='patch size', default=60, type=int)
    parser.add_argument('--num_pos', help='the number of positives to crop', default=1000, type=int)
    parser.add_argument('--num_neg', help='the number of negatives to crop', default=1000, type=int)
    parser.add_argument('--shuffle', help='shuffle the dataset', default=False, type=bool)
    parser.add_argument('--overlap', help='IoU between a negative and pos and existing neg', default=0.3, type=float)
    parser.add_argument('--ignore_existed', help='ignore existed data or not', default=True, type=bool)
    args = parser.parse_args()
    return args


# --image_dir ./data/FDDB/originalPics --annotation_dir ./data/FDDB/FDDB-folds --which_fold 1,2,3 --save_dir ./data/cache

if __name__ == '__main__':
    args = parse_args()
    create_datasets_FDDB(args.image_dir, args.annotation_dir, args.which_folds,
                         args.save_dir, args.is_training, args.use_color, args.patch_size,
                         args.num_pos, args.num_neg, args.shuffle, args.overlap, args.ignore_existed)


