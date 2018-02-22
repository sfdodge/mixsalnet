"""
Code to load salicon database into an h5 file
"""
import scipy.io
import numpy as np
import h5py
import os
import glob
import samlib
import utilities
import cv2
from sklearn.model_selection import StratifiedKFold


TYPES = ['Action','Art','Cartoon','Indoor','Jumbled','LowResolution','Object','OutdoorNatural','Random','Sketch','Affective','BlackWhite','Fractal','Inverted','LineDrawing','Noisy','OutdoorManMade','Pattern','Satelite','Social',]


class DB():
    def __init__(self, dbname, basepath, im_shape, gt_shape, gen=True, seed=0):
        self.path = basepath
        self.dbname = dbname
        self.im_shape = im_shape
        self.gt_shape = gt_shape

        if gen:
            self.generate_db()
        else:
            self.db = h5py.File(dbname, 'r')

        # create split
        self.train_test_split(seed)

    def generate_db(self):
        """
        This is the main function to load the database
        """
        print "Generating database..."

        # delete if db already exists
        if os.path.exists(self.dbname):
            os.remove(self.dbname)

        # start database
        self.db = h5py.File(self.dbname, 'w')

        names = glob.glob(self.path + 'images/*/*.jpg')

        im_dir = self.path + 'images/'
        fix_dir = self.path + 'fixation/'
        dense_dir = self.path + 'FIXATIONMAPS/'
        num = len(names)

        self.db['images'] = np.empty((num, 3, self.im_shape[0], self.im_shape[1]),
                                     dtype=np.float32)
        self.db['fixations'] = np.empty((num, self.gt_shape[0], self.gt_shape[1]),
                                    dtype=np.float32)
        self.db['maps'] = np.empty((num, self.gt_shape[0], self.gt_shape[1]),
                                    dtype=np.float32)
        self.db['class'] = np.empty(num, dtype=np.uint8) # the class of each box
        self.db['image_names'] = np.empty((num,1),dtype='|S150') # the image names

        pbar = samlib.misc.pbar('Generate database:')
        for i, name in enumerate(pbar(names)):
            im, fix, dense = self.loadIm(name, im_dir, fix_dir, dense_dir)

            self.db['images'][i] = im
            self.db['fixations'][i] = fix
            self.db['maps'][i] = dense

            split = name.split('/')
            self.db['class'][i] = TYPES.index(split[-2])
            self.db['image_names'][i] = name

    def loadIm(self, name, im_dir, fix_dir, dense_dir, EXT='jpg'):
        """
        Load a single image and ground truth (perform all preprocessing)
        """
        im = self.preprocess_image(name)

        # next is the ground truth fixation points
        split = name.split(os.sep)
        fix_path = fix_dir + os.path.join(*split[-2:])[:-4] + '.mat'
        mat = scipy.io.loadmat(fix_path)['fixLocs']

        inds = np.nonzero(mat)

        scaleRow = float(self.gt_shape[0]) / float(mat.shape[0])
        scaleCol = float(self.gt_shape[1]) / float(mat.shape[1])

        rows = inds[0] * scaleRow
        cols = inds[1] * scaleCol

        newMat = np.zeros(self.gt_shape)

        for r, c in zip(rows, cols):
            newMat[r, c] += 1

        newMat = newMat / np.sum(newMat)

        fix_pts = newMat

        # finally we have the actual fixation maps
        fix_path = dense_dir + os.path.join(*split[-2:])
        fix_map = self.preprocess_map(fix_path)

        return im, fix_pts, fix_map

    def train_test_split(self, seed):
        """
        get a random train and test split
        """
        n = self.db['images'].shape[0]

        np.random.seed(seed)

        # split into folds
        all_cat = self.db['class'][:]
        kf = StratifiedKFold(n_splits=5)
        splits = [(tr, te) for tr,te in kf.split(np.ones((n,1)), all_cat)]
        trainval_inds = splits[seed][0]
        test_inds = splits[seed][1]
        train_inds, val_inds = utilities.randsplit(all_cat[trainval_inds], 0.75)
        train_inds = trainval_inds[train_inds]
        val_inds = trainval_inds[val_inds]

        self.train_inds = np.sort(train_inds)
        self.test_inds = np.sort(test_inds)
        self.val_inds = np.sort(val_inds)

        self.train_size = len(self.train_inds)
        self.val_size = len(self.val_inds)
        self.test_size = len(self.test_inds)

        # recompute center bias based on training data
        allFix = np.zeros(self.gt_shape)
        for y in self.db['fixations'][self.train_inds, :]:
            y2 = np.reshape(y, self.gt_shape)
            allFix = allFix + y2

        # save all fixations (for modeling center bias)
        allFix = allFix / np.sum(allFix)  # already normalize for theano snss
        self.allFix = allFix

    def get_size(self, db_type, cat='all'):
        inds = self.get_inds(db_type, cat)
        return len(inds)

    def close(self):
        # close database
        self.db.close()

    def __getitem__(self, key):
        return self.db[key]

    def get_inds(self, db_type, cat='all'):
        """
        returns indices for different sets
        """
        if db_type == 'train':
            inds = self.train_inds
        elif db_type == 'val':
            inds = self.val_inds
        elif db_type == 'test':
            inds = self.test_inds

        if cat != 'all':
            cat_ind = TYPES.index(cat)
            cats = self.db['class'][:][inds]
            valid = cats == cat_ind
            inds = inds[valid]

        return inds

    def preprocess_image(self, path):
        original_image = cv2.imread(path)
        padded_image = utilities.padding(original_image,
                                         self.im_shape[0], self.im_shape[1], 3)
        im = padded_image.astype('float32')

        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))

        return im

    def preprocess_map(self, path):
        original_image = cv2.imread(path, 0)
        padded_image = utilities.padding(original_image,
                                         self.gt_shape[0], self.gt_shape[1], 1)
        im = padded_image.astype('float32')
        im /= 255.0

        return im
