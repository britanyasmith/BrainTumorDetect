import tensorflow as tf 
from d2l import tensorflow as d2l 



#d2l.classifier 

class Model(d2l.classifier): 
    def __init__(self, lr = 0.1, num_classes=10):
        #super().__init__()  #
        #self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'), 
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2), 
            tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'), 
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2), 
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'), 
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'), 
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'), 
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(num_classes)
        ])  #Network creation



class LitModel(LightningModule):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = nn.Linear(self.hparams.in_dim, self.hparams.out_dim)