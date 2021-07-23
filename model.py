"""
    model.py: CNN神经网络模型
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def cnn_inference(images, batch_size, n_classes):
    """
        输入：
            images：队列中取的一批图片, 具体为：4D tensor [batch_size, width, height, 3]
            batch_size：每个批次的大小
            n_classes：n分类（这里是二分类，猫或狗）
        返回：
            softmax_linear：表示图片列表中的每张图片分别是猫或狗的预测概率（即：神经网络计算得到的输出值）。
                            例如: [[0.459, 0.541], ..., [0.892, 0.108]],
                            一个数值代表属于猫的概率，一个数值代表属于狗的概率，两者的和为1。
    """

    # TensorFlow中的变量作用域机制：
    #       tf.variable_scope(<scope_name>): 指定命名空间
    #       tf.get_variable(<name>, <shape>, <dtype>, <initializer>): 创建一个变量

    # 第一层的卷积层conv1，卷积核(weights)的大小是 3*3, 输入的channel(管道数/深度)为3, 共有16个
    with tf.variable_scope('conv1') as scope:
        # tf.truncated_normal_initializer():weights初始化生成截断正态分布的随机数，stddev标准差
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))   # 初始化为常数，通常偏置项biases就是用它初始化的

        # strides = [1, y_movement, x_movement, 1], 每个维度的滑动窗口的步幅,一般首末位置固定都为1
        # padding = 'SAME', 是考虑边界, 不足时用0去填充周围
        # padding = 'VALID', 不考虑边界, 不足时舍弃不填充周围
        # 参考：https://blog.csdn.net/qq_36201400/article/details/108454066
        # 输入的images是[16,208,208,3], 即16张 208*208 大小的图片, 图像通道数是3
        # weights(卷积核)的大小是 3*3, 数量为16
        # strides(滑动步长)是[1,1,1,], 即卷积核在图片上卷积时分别向x、y方向移动为1个单位
        # 由于padding='SAME'考虑边界，最后得到16张图且每张图得到16个 208*208 的feature map(特征图)
        # conv(最后输出的结果)是shape为[16,208,208,16]的4维张量(矩阵/向量)
        # 用weights卷积核对images图片进行卷积
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)      # 加入偏差，biases向量与矩阵的每一行进行相加, shape不变
        conv1 = tf.nn.relu(pre_activation, name='conv1')   # 在conv1的命名空间里，用relu激活函数非线性化处理

    # 第一层的池化层pool1和规范化norm1(特征缩放）
    with tf.variable_scope('pooling1_lrn') as scope:
        # 对conv1池化得到feature map
        # 参考：https://blog.csdn.net/qq_22968719/article/details/88318626
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        # lrn()：局部响应归一化, 一种防止过拟合的方法, 增强了模型的泛化能力，
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # 第二层的卷积层cov2，卷积核(weights)的大小是 3*3, 输入的channel(管道数/深度)为16, 共有16个
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],  # 这里的第三位数字16需要等于上一层的tensor维度
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # 第二层的池化层pool2和规范化norm2(特征缩放）
    with tf.variable_scope('pooling2_lrn') as scope:
        # 这里选择了先规范化再池化
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')

    # 第三层为全连接层local3
    # 连接所有的特征, 将输出值给分类器 (将特征映射到样本标记空间), 该层映射出256个输出
    with tf.variable_scope('local3') as scope:
        # 将pool2张量铺平, 再把维度调整成shape(shape里的-1, 程序运行时会自动计算填充)
        # 参考：https://blog.csdn.net/csdn0006/article/details/106238909/
        reshape = tf.reshape(pool2, shape=[batch_size, -1])

        dim = reshape.get_shape()[1].value            # 获取reshape后的列数
        weights = tf.get_variable('weights',
                                  shape=[dim, 256],   # 连接256个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 矩阵相乘再加上biases，用relu激活函数非线性化处理
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='local3')

    # 第四层为全连接层local4
    # 连接所有的特征, 将输出值给分类器 (将特征映射到样本标记空间), 该层映射出512个输出
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[256, 512],  # 再连接512个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 矩阵相乘再加上biases，用relu激活函数非线性化处理
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # 第五层为输出层(回归层): softmax_linear
    # 将前面的全连接层的输出，做一个线性回归，计算出每一类的得分，在这里是2类，所以这个层输出的是两个得分。
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights',
                                  shape=[512, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        # softmax_linear的行数=local4的行数，列数=weights的列数=bias的行数=需要分类的个数
        # 经过softmax函数用于分类过程中，它将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解
        # 这里local4与weights矩阵相乘，再矩阵相加biases
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    # 这里没做归一化和交叉熵。真正的softmax函数放在下面的losses()里面和交叉熵结合在一起了，这样可以提高运算速度。
    # 图片列表中的每张图片分别被每个分类取到的概率，
    return softmax_linear


def losses(logits, labels):
    """
        输入：
            logits: 经过cnn_inference得到的神经网络输出值（图片列表中每张图片分别是猫或狗的预测概率）
            labels: 图片对应的标签（即：真实值。用于与logits预测值进行对比得到loss）
        返回：
            loss： 损失值（label真实值与神经网络输出预测值之间的误差）
    """
    with tf.variable_scope('loss') as scope:
        # label与神经网络输出层的输出结果做对比，得到损失值（这做了归一化和交叉熵处理）
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='loss_per_eg')
        loss = tf.reduce_mean(cross_entropy, name='loss')  # 求得batch的平均loss（每批有16张图）
    return loss


def training(loss, learning_rate):
    """
        输入：
            loss: 训练中得到的损失值
            learning_rate：学习率
        返回：
            train_op: 训练的最优值。训练op，这个参数要输入sess.run中让模型去训练。
    """
    with tf.name_scope('optimizer'):
        # tf.train.AdamOptimizer():
        # 除了利用反向传播算法对权重和偏置项进行修正外，也在运行中不断修正学习率。
        # 根据其损失量学习自适应，损失量大则学习率越大，进行修正的幅度也越大;
        #                     损失量小则学习率越小，进行修正的幅度也越小，但是不会超过自己所设定的学习率。
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)    # 使用AdamOptimizer优化器来使loss朝着变小的方向优化

        global_step = tf.Variable(0, name='global_step', trainable=False)  # 全局步数赋值为0

        # loss：即最小化的目标变量，一般就是训练的目标函数，均方差或者交叉熵
        # global_step：梯度下降一次加1，一般用于记录迭代优化的次数，主要用于参数输出和保存
        train_op = optimizer.minimize(loss, global_step=global_step)   # 以最大限度地最小化loss

    return train_op


def evaluation(logits, labels):
    """
        输入：
            logits: 经过cnn_inference得到的神经网络输出值（图片列表中每张图片分别是猫或狗的预测概率）
            labels: 图片对应的标签（真实值，0或1）
        返回：
            accuracy：准确率（当前step的平均准确率。即：这些batch中多少张图片被正确分类了）
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)   # 用法参考：https://www.cnblogs.com/logo-88/p/9099383.html
        correct = tf.cast(correct, tf.float16)        # 转换格式为浮点数
        accuracy = tf.reduce_mean(correct)            # 计算当前批的平均准确率
    return accuracy
