from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import datetime

from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras import Model
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train
x_train.min(), x_train.max()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train.min(), x_train.max()
# Add a "channels" dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
x_train.shape  # x_test.shape
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
train_ds
class MyModel(Model):   # tf.keras.Model class
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, (3,3), activation='relu')  # tf.keras.layers.Conv2D()
        self.batchnorm1 = BatchNormalization()
        self.conv2 = Conv2D(64,(3,3))  # tf.keras.layers.Conv2D()
        self.batchnorm3 = BatchNormalization()
        self.conv3 = Conv2D(128,(3,3))  # tf.keras.layers.Conv2D()
        self.batchnorm4 = BatchNormalization()
        self.maxpool = MaxPooling2D((2,2))
        self.flatten = Flatten()                 # tf.keras.layers.Flatten()
        self.d1 = Dense(128, activation='relu')  # tf.keras.layers.Dense()
        self.batchnorm2 = BatchNormalization()   # tf.keras.layers.BatchNormalization()
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.conv3(x)
        x = self.batchnorm4(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.batchnorm2(x)
        return self.d2(x)

# Create an instance of the model
model = MyModel()
model
#  < tf.keras.losses.SparseCategoricalCrossentropy >
#    from_logits: Whether y_pred is expected to be a logits tensor. 
#                 By default, we assume that y_pred encodes 
#                 a probability distribution. 
#   [Note]: Using from_logits=True may be more numerically stable.

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
@tf.function
def train_step(images, labels):    # images : x_train , labels : y_train
    ## ----------------------------------------------------------------------
    ##  Forward propagation - 
    ##    tf.GradientTape()可以用在 training loop 裡，記錄並建構正向傳播的計算圖
    ## ----------------------------------------------------------------------
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)

    ## ----------------------------------------------------------------------
    ##  Backpropagation - 
    ##    在完成“記錄”後，tf.GradientTape() 的 tape 物件則呼叫 gradient()方法，
    ##    並傳入損失值 (loss score) 和模型可訓練的參數。 [from Ref 3.]
    ## ----------------------------------------------------------------------
    gradients = tape.gradient(loss, model.trainable_variables)
    
    ## ----------------------------------------------------------------------
    ##  Parameters' update - 
    ##    一旦計算出了梯度後，立即呼叫 optimizer.apply_gradients() 方法，
    ##    傳入一個 list of tuple，每一個 tuple 的第二個則是參數變數，
    ##    而第一個變數為針對該參數所計算出的梯度。 [from Ref 3.]
    ## ----------------------------------------------------------------------    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_directory = 'logs/CNN_MNIST_BN/'
train_log_dir = log_directory + current_time + '/train'
test_log_dir = log_directory + current_time + '/test'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
EPOCHS = 10

for epoch in range(EPOCHS):
    import time
    start=time.time()
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)
        
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
        
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
    
    end=time.time()
    
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100),", Running time: ", str(end-start), "seconds")