import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm

tf.logging.set_verbosity(tf.logging.ERROR)
# 由于版本问题，可以忽略 tensor 的警告

# 设置
Learning_rate = 1e-4  # 学习率
Training_iterations = 2500  # 迭代次数
Dropout = 0.5  # 每次杀死50%的神经元，防止过拟合
Batch_size = 50  # 每次迭代50张图像
Validation_size = 2000  # 验证集
Image_to_display = 10  # 输出10种类型

# 读取训练集
data = pd.read_csv('mnist_train.csv')

print('data({0[0]}, {0[1]})'.format(data.shape))
print(data.head())
print('\n')

# 数据简单预处理
images = data.iloc[:, 1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)  # 归一化数据

print('images({0[0]}, {0[1]})'.format(images.shape))
print('\n')

# 图像像素
image_size = images.shape[1]
print('image_size => {0}'.format(image_size))
print('\n')

# 图像的宽和高
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))
print('\n')


def display(img):
    one_image = img.reshape(image_width, image_height)
    # 转换图像格式

    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()


display(images[Image_to_display])

# 查看 label
labels_flat = data.iloc[:, 0].values.ravel()

print('labels_flat({0})'.format(len(labels_flat)))
print('labels_flat[{0}] => {1}'.format(Image_to_display, labels_flat[Image_to_display]))

labels_count = np.unique(labels_flat).shape[0]

print('labels_count => {0}'.format(labels_count))
print('\n')


# 对数据进行 one-hot 编码
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))

    # flat就是相当于变成一维数组,再读取
    # ravel将多维数组转化为一维，返回一个连续的平整的数组。
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

# print('labels({0[0]}, {0[1]})'.format(labels.shape))
# print('labels[{0}] => {1}'.format(Image_to_display, labels[Image_to_display]))


# 数据集的切分
validation_images = images[:2000]
validation_labels = labels[:2000]

train_images = images[2000:]
train_labels = labels[2000:]

print('train_images({0[0]}, {0[1]})'.format(train_images.shape))
print('train_labels({0[0]}, {0[1]})'.format(train_labels.shape))
print('\n')


# 建立神经网络
# 权重初始化
def weight_variable(shape):
    # 注：tensor 需要初始化，需要将数据转换为 tensor 支持的格式
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 补充层
def con2d(x, W):
    # x 指输入；W 指CNN的卷积核；strides 卷积时在图像上每一维的步长，一般首尾为1，中间为自定义的步长
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化
def max_pool_2x2(x):
    # x 指池化输入，ksize 为池化窗口的大小，strides 为窗口在每一个维度上滑动的步长
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义占位符
x = tf.placeholder(tf.float32, shape=[None, image_size])
y_ = tf.placeholder(tf.float32, shape=[None, labels_count])

# 第一层卷积神经网络，选择一个5*5的窗口，初始图像为28*28*1，将图像分为32个特征图
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

image = tf.reshape(x, [-1, image_width, image_height, 1])
print(image.get_shape())
print('\n')

h_conv1 = tf.nn.relu(con2d(image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print(h_conv1.get_shape())
print(h_pool1.get_shape())
print('\n')

# 第二层卷积神经网络
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(con2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print(h_conv2.get_shape())
print(h_pool2.get_shape())
print('\n')

# 定义全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print(h_pool2_flat.get_shape())
print(h_fc1.get_shape())
print('\n')

# 防止过拟合，随机杀死一些神经元
keep_prob = tf.placeholder('float')  # 保存率
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])

# 使用softmax来得到各分类的预测分数
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print(y.get_shape())
print('\n')

# 损失函数，这里使用了交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 优化函数
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(cross_entropy)

# 评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# 预测函数
predict = tf.argmax(y, 1)

# 训练，验证，预测
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]


# 按 batch 迭代数据
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # 当所有训练数据都已被使用时，它会被随机重新排序
    if index_in_epoch > num_examples:
        epochs_completed += 1

        # 冲洗数据
        perm = np.arange(num_examples)
        np.random.shuffle(perm)

        train_images = train_images[perm]
        train_labels = train_labels[perm]

        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

# 变量可视化
train_accuracies = []
validation_accuracies = []
x_range = []
display_step = 1

# 迭代多次，需要 next_batch
for i in range(Training_iterations):

    # 获得新的批次
    batch_xs, batch_ys = next_batch(Batch_size)

    # 判断每一步的进度
    if i % display_step == 0 or (i + 1) == Training_iterations:

        # 传入 x,y_
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})

        if (Validation_size):
            validation_accuracy = accuracy.eval(feed_dict={x: validation_images[0: Batch_size],
                                                           y_: validation_labels[0: Batch_size],
                                                           keep_prob: 1.0})

            print('train_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (train_accuracy,
                                                                                       validation_accuracy, i))
            validation_accuracies.append(validation_accuracy)

        else:
            print('training_accuracy => %.4f for step %d' % (train_accuracy, i))

        train_accuracies.append(train_accuracy)
        x_range.append(i)

        # 增加显示步骤
        if i % (display_step * 10) == 0 and i:
            display_step *= 10

    # 批量训练
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: Dropout})

# 检测 train 和 validation 的准确度
if (Validation_size):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                   y_: validation_labels,
                                                   keep_prob: 1.0})

    plt.plot(x_range, train_accuracies, '-b', label='Training')
    plt.plot(x_range, validation_accuracies, '-g', label='Validation')

    plt.legend(loc='lower right', frameon=False)
    plt.ylim(top=1.1, bottom=0.7)

    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()
