"""
Main training code of mixnet model
"""
import numpy as np
np.random.seed(0)
from keras.callbacks import EarlyStopping
from model import build_model, sal_loss
from keras.utils import np_utils
import utilities
import keras

# my functions
from db import DB

# parameters
from param import *


def main():
    """
    Main training function
    """
    print "using seed " + str(SEED)
    print "=" * 50
    # load database
    db = DB(BASE_PATH + 'db.h5',
            BASE_PATH,
            IM_SHAPE,
            GT_SHAPE,
            gen=GEN_DB,
            seed=SEED)

    # Create neural network model
    print("Building model...")
    model = build_model()

    model.load_weights(LOAD_NAME, by_name=True)

    model.compile(
        'adadelta',
        {'output_ml_net': sal_loss,
         'cat_soft': 'categorical_crossentropy'},
        metrics={'cat_soft': 'accuracy'},
        loss_weights={'output_ml_net': 10.0, 'cat_soft': 1.0})

    EARLY_STOP = 10

    # do training
    print("Training...")
    history = keras.callbacks.History()
    model.fit_generator(
        generator(db, 'train', BATCHSIZE, shuffle=True),
        db.get_size('train'), nb_epoch=NB_EPOCH,
        validation_data=generator(db, 'val', BATCHSIZE, augment=False),
        nb_val_samples=db.get_size('val'),
        callbacks=[EarlyStopping(patience=EARLY_STOP, monitor='val_loss'),
                   history],
        verbose=1)

    # close dataset
    db.close()

    # save model
    model.save(MODEL_DIR + MODEL_NAME + '.h5')

    # send email
    samlib.misc.sendmail('Saliency done')


def generator(db, dbtype, batchsize, shuffle=False, augment=True):
    inputs = db['images']
    classes = db['class']
    targets = db['maps']
    indices = db.get_inds(dbtype)

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
                sortind = np.sort(excerpt).tolist()
            else:
                sortind = slice(start_idx, start_idx + batchsize)

            # take subset (should be numpy array)
            batch_inputs = inputs[sortind]
            batch_targets = targets[sortind]
            batch_class = classes[sortind]
            batch_class = np_utils.to_categorical(batch_class, 20)

            if AUGMENT_DATA and augment:
                in_shape =  batch_inputs.shape[0] # number of images in batch
                ind = np.random.choice(in_shape, in_shape / 2, replace=False)

                # flip
                batch_inputs[ind] = utilities.flip(batch_inputs, ind)
                batch_targets[ind] = utilities.flip(batch_targets, ind)

            batch_targets = np.expand_dims(batch_targets, axis=1)
            yield batch_inputs, [batch_targets, batch_class]

# cross validation
for i in range(0, 5):
    SEED = i
    MODEL_NAME = 'mixnet_' + str(SEED)

    LOAD_NAME = 'models/mlnet_salicon.h5'
    main()
