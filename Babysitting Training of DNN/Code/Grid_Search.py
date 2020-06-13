import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def get_coordinates(line):
    line = line.replace('\n','')   
    line = line.split(' ')
    line.remove('')
    line = [float(i) for i in line]    
    [major_radius, minor_radius, angle, center_x, center_y, _] = line
 
    x = int(center_x - minor_radius)
    y = int(center_y - major_radius)
    width = int(2 * minor_radius)
    height = int(2 * major_radius)
    
    return [x, y, width, height]


dataset_path = 'E:\Sem 4\CV\Projects\Project03\FDDB'

X_train = []
Y_train = []
img_size = 200

for index in range(1, 11):
    file_name = 'FDDB-fold-' + str(index).zfill(2) + '-ellipseList.txt'
    file_path = dataset_path + '/FDDB-folds/' + file_name
    fp = open(file_path, 'r')
    lines = fp.readlines()

    for i in range (len(lines)):
        line = lines[i].replace('\n', '')
        
        if(line == '1'):
            image_path = lines[i-1].replace('\n', '') + '.jpg'
            image_path = dataset_path + '/originalPics/' + image_path
            line_coords = lines[i+1]
            [x, y, width, height] = get_coordinates(line_coords)
            img = Image.open(image_path)
            [W, H] = img.size
            
            img_face = img.crop((x, y, x+width, y+height))
            img_face = img_face.resize((img_size, img_size))
            img_face = np.array(img_face)
              
            img_nonface = img.crop((x+width, y+height, W, H))
            img_nonface = img_nonface.resize((img_size, img_size))  
            img_nonface = np.array(img_nonface)
            
            if(img_face.shape == (img_size, img_size, 3)):
               X_train.append(img_face)
               Y_train.append([1,0])
               
               X_train.append(img_nonface) 
               Y_train.append([0,1])
               
               
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)  

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

epochs = 10
batch_size = 32
n_classes = 2                
drop_out = 0.2               
n_channels = 3   
            
len_train_set = X_train.shape[0]
len_test_set = X_test.shape[0]


def weight(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2d(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


print("Building model")

x = tf.placeholder(tf.float32, [None, img_size*img_size])
x = tf.reshape(x, [-1, img_size, img_size, n_channels])
y = tf.placeholder(tf.float32, shape=[None, 2])

prob = tf.placeholder(tf.float32)

W_conv1 = weight([5, 5, n_channels, 16])
b_conv1 = bias([16])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2d(h_conv1)

W_conv2 = weight([5, 5, 16, 16])
b_conv2 = bias([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2d(h_conv2)

flattened = tf.reshape(h_pool2, [-1, 50*50*16])

W_fc1 = weight([50*50*16, 64])
b_fc1 = bias([64])
h_fc1 = tf.nn.relu(tf.matmul(flattened, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, 1-drop_out)

W_fc2 = weight([64, n_classes])
b_fc2 = bias([n_classes])

y_ = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

prediction = tf.nn.softmax(y_)

total_batches_train = int(len_train_set/batch_size)
total_batches_test = int(len_test_set/batch_size)

sess = tf.Session()

for count in range(10):
    learning_rate = np.random.uniform(0.0003, 0.001) #1e-6  #np.random.uniform(0.000670, 0.000960)
    reg = np.random.uniform(0.004, 0.03) #0.00001 #np.random.uniform(0.004194, 0.020000)
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = y_)) + \
                    reg*(tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + \
                                    tf.nn.l2_loss(W_fc2))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    train_acc_arr = []
    val_acc_arr = []
    cost_arr = []    
    epoch_arr = []
    
    print('Learning Rate:', learning_rate, '   Regularization:', reg)
    
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batches_train):
            batch_x = X_train[i * batch_size: (i+1) * batch_size]        
            batch_y = Y_train[i * batch_size: (i+1) * batch_size] 
            
            _, c = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y: batch_y, prob: 1-drop_out})
            avg_cost += c/total_batches_train
        
        # Training Accuracy
        acc_train = 0
        for i in range(total_batches_train):
            batch_x = X_train[i * batch_size: (i+1) * batch_size]        
            batch_y = Y_train[i * batch_size: (i+1) * batch_size] 
            acc_train += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, prob:1.0})/total_batches_train
        
        # Validation Accuracy 
        acc_val = 0
        for i in range(total_batches_test):
            batch_x = X_test[i * batch_size: (i+1) * batch_size]        
            batch_y = Y_test[i * batch_size: (i+1) * batch_size] 
            acc_val += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, prob:1.0})/total_batches_test
        
        epoch_arr.append(epoch+1) 
        cost_arr.append(avg_cost)    
        train_acc_arr.append(acc_train)
        val_acc_arr.append(acc_val)

        print("cost:", '{0:.6f}'.format(avg_cost), "  train_acc:", '{0:.6f}'.format(acc_train),\
              "  val_acc:", '{0:.6f}'.format(acc_val), "  learning_rate:", '{0:.6f}'.format(learning_rate),\
              "  regularization:", '{0:.6f}'.format(reg))
