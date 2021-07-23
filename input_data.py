"""
    input_data.py: 读取训练数据
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os


def get_files(file_dir):
    """
        输入：
            file_dir：存放训练图片的文件地址
        返回:
            image_list：乱序后的图片路径列表
            label_list：乱序后的标签(相对应图片)列表
    """
    # 建立空列表
    cats = []           # 存放是猫的图片路径地址
    label_cats = []     # 对应猫图片的标签
    dogs = []           # 存放是猫的图片路径地址
    label_dogs = []     # 对应狗图片的标签

    # 从file_dir路径下读取数据，存入空列表中
    for file in os.listdir(file_dir):     # file就是要读取的图片带后缀的文件名
        name = file.split(sep='.')        # 图片格式是cat.1.jpg / dog.2.jpg, 处理后name为[cat, 1, jpg]
        if name[0] == 'cat':              # name[0]获取图片名
            cats.append(file_dir + file)  # 若是cat，则将该图片路径地址添加到cats数组里
            label_cats.append(0)          # 并且对应的label_cats添加0标签 （这里记作：0为猫，1为狗）
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)          # 注意：这里添加进的标签是字符串格式，后面会转成int类型

    # print('There are %d cats\nThere are %d dogs' % (len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))               # 在水平方向平铺合成一个行向量，即两个数组的拼接
    label_list = np.hstack((label_cats, label_dogs))   # 这里把猫狗图片及标签合并分别存在image_list和label_list
    temp = np.array([image_list, label_list])  # 生成一个2 X 25000的数组，即2行、25000列
    temp = temp.transpose()                    # 转置向量，大小变成25000 X 2
    np.random.shuffle(temp)                    # 乱序，打乱这25000行排列的顺序

    image_list = list(temp[:, 0])              # 所有行，列=0（选中所有猫狗图片路径地址），即重新存入乱序后的猫狗图片路径
    label_list = list(temp[:, 1])              # 所有行，列=1（选中所有猫狗图片对应的标签），即重新存入乱序后的对应标签
    label_list = [int(float(i)) for i in label_list]  # 把标签列表转化为int类型（用列表解析式迭代，相当于精简的for循环）

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
        输入：
            image,label：要生成batch的图像和标签
            image_W，image_H: 图像的宽度和高度
            batch_size: 每个batch（小批次）有多少张图片数据
            capacity: 队列的最大容量
        返回：
            image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
            label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
    image = tf.cast(image, tf.string)   # 将列表转换成tf能够识别的格式
    label = tf.cast(label, tf.int32)

    # 队列的理解：
    #     每次训练时，从队列中取一个batch送到网络进行训练，然后又有新的图片从训练库中注入队列，这样循环往复。
    #     队列相当于起到了训练库到网络模型间数据管道的作用，训练数据通过队列送入网络。
    input_queue = tf.train.slice_input_producer([image, label])   # 生成队列(牵扯到线程概念，便于batch训练), 将image和label传入
    # input_queue = tf.optimizers.slice_input_producer([image, label])   # Tensorflow 2.0版本

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])              # 图像的读取需要tf.read_file(), 标签则可以直接赋值。
    image = tf.image.decode_jpeg(image_contents, channels=3)   # 使用JPEG的格式解码从而得到图像对应的三维矩阵。
    # 注意：这里image解码出来的数据类型是uint8, 之后模型卷积层里面conv2d()要求传入数据为float32类型

    # 图片数据预处理：统一图片大小(缩小图片) + 标准化处理
    # ResizeMethod.NEAREST_NEIGHBOR：最近邻插值法，将变换后的图像中的原像素点最邻近像素的灰度值赋给原像素点的方法，返回图像张量dtype与所传入的相同。
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)                  # 将image转换成float32类型
    image = tf.image.per_image_standardization(image)   # 图片标准化处理，加速神经网络的训练

    # 按顺序读取队列中的数据
    image_batch, label_batch = tf.train.batch([image, label],          # 进队列的tensor列表数据
                                              batch_size=batch_size,   # 设置每次从队列中获取出队数据的数量
                                              num_threads=64,          # 涉及到线程，配合队列
                                              capacity=capacity)       # 用来设置队列中元素的最大数量

    return image_batch, label_batch