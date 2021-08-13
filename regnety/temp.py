import tensorflow as tf
from regnety.models import RegNetY

randx = tf.random.uniform((20, 32,32 ,3))
randy = tf.random.uniform((20, 5))

model = RegNetY("200mf")

model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(randx, randy, epochs=10)

