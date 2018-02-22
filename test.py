"""
Test the model
"""
import numpy as np
np.random.seed(0)
from db import DB

from model import build_model
# from baseline_model import build_model

import samlib.misc.pbar
import os
import scipy
from utilities import postprocess_predictions

from param import *

CAT_NAMES = ['Action','Art','Cartoon','Indoor','Jumbled','LowResolution','Object','OutdoorNatural','Random','Sketch','Affective','BlackWhite','Fractal','Inverted','LineDrawing','Noisy','OutdoorManMade','Pattern','Satelite','Social',]

OUTPUT_FOLDER = 'Outputs/mixnet'


def main(seed=0):
    # model_weights = ('models/mlnet_finetune_' + str(seed) + '.h5',)
    model_weights = ('models/mixnet_' + str(seed) + '.h5',)

    # load model
    models = []
    for mw in model_weights:
        model = build_model()

        model.load_weights(mw)
        models.append(model)

    # load database
    GEN_DB = False
    db = DB(BASE_PATH + 'db.h5',
            BASE_PATH,
            IM_SHAPE,
            GT_SHAPE,
            gen=GEN_DB,
            seed=seed)

    test_inds = db.get_inds('test')

    inputs = db['images']
    fnames = db['image_names']
    bar = samlib.misc.pbar()
    for i in bar(test_inds):
        fname = fnames[i]

        pred_maps = []
        for model in models:
            pred_map = model.predict(inputs[i:i+1])

            if isinstance(pred_map, (list, tuple)):
                pred_map = pred_map[0]

            pred_maps.append(pred_map)

        pred_map = np.mean(np.stack(pred_maps), axis=0)
        pred_map = pred_map[0, 0]

        # save
        pred_map = postprocess_predictions(pred_map, 1080, 1920)
        split = fname[0].split('/')
        odir = os.path.join(OUTPUT_FOLDER, split[-2])
        if not os.path.exists(odir):
            os.makedirs(odir)
        scipy.misc.imsave(os.path.join(odir, split[-1][:-3] + 'png'), pred_map)

    db.close()

for seed in range(0, 5):
    print "Seed = " + str(seed)
    main(seed)
