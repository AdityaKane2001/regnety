import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from regnety.regnety.models import RegNetY

model = RegNetY('200mf')

model.build((128,224,224,3))

model.plot_model()
