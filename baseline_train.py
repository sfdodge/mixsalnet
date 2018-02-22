"""
Finetune mlnet model on CAT2000 dataset
"""
import numpy as np
np.random.seed(0)
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from baseline_model import build_model, loss
import utilities

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
    model = build_model(use_pretrained='mlnet')
    sgd = SGD(lr=LEARNING_RATE, decay=LEARNING_DECAY, momentum=MOMENTUM, nesterov=True)
    model.compile(sgd, loss)

    # do training
    print("Training...")
    model.fit_generator(
        generator(db, 'train', BATCHSIZE, shuffle=True),
        db.get_size('train', CATEGORY), nb_epoch=NB_EPOCH,
        validation_data=generator(db, 'val', BATCHSIZE, augment=False),
        nb_val_samples=db.get_size('val', CATEGORY),
        callbacks=[EarlyStopping(patience=EARLY_STOP), ],
        verbose=1)

    # close dataset
    db.close()

    # save model
    model.save(MODEL_DIR + MODEL_NAME + '.h5')


def generator(db, dbtype, batchsize, shuffle=False, augment=True):
    inputs = db['images']
    targets = db['maps']
    indices = db.get_inds(dbtype, cat=CATEGORY)

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

            if AUGMENT_DATA and augment:
                in_shape = batch_inputs.shape[0] # number of images in batch
                ind = np.random.choice(in_shape, in_shape / 2, replace=False)

                # flip
                batch_inputs[ind] = utilities.flip(batch_inputs, ind)
                batch_targets[ind] = utilities.flip(batch_targets, ind)

            batch_targets = np.expand_dims(batch_targets, axis=1)
            yield batch_inputs, batch_targets

# do the 5 fold cross validation
for i in range(0, 5):
    SEED = i
    MODEL_NAME = 'mlnet_finetune_' + str(SEED)
    main()
