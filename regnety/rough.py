from regnety.regnety.models import RegNetY
import tensorflow as tf
import regnety.regnety.utils.model_utils as mutil


tf.keras.backend.clear_session()
model = RegNetY('200mf')

print(model.inputs)
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = 'accuracy')
print(mutil.get_flops(model))