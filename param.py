############################################################
# Parameters
############################################################
# Training parameters
# --------------------
CATEGORY = "all"
BATCHSIZE = 8
NB_EPOCH = 100
AUGMENT_DATA = True # for now this is just flipping

# Learning parameters
LEARNING_RATE = 1e-3
LEARNING_DECAY = 0.0005
MOMENTUM = 0.9
EARLY_STOP = 10 # stop if the validation error is not improving

# Database parameters
GEN_DB = False
IM_SHAPE = (480, 640)
GT_SHAPE = (480/8, 640/8)
DATABASE  = 'cat2000' 
BASE_PATH = '/home/sfdodge/Documents/Datasets/CAT2000/'
LOAD_TYPE = DATABASE + 'vgg'    

# Saving Parameters
# --------------------
MODEL_DIR      = 'models/'
MODEL_NAME     = 'mixnet'
