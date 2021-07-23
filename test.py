"""
    test.py: 用训练好的模型对随机一张图片进行猫狗预测
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import matplotlib.pyplot as plt
import input_data
import model
import numpy as np


def get_one_image(img_list):
    """
        输入：
            img_list：图片路径列表
        返回：
            image：从图片路径列表中随机挑选的一张图片
    """
    n = len(img_list)                  # 获取文件夹下图片的总数
    ind = np.random.randint(0, n)      # 从 0~n 中随机选取下标
    img_dir = img_list[ind]            # 根据下标得到一张随机图片的路径

    image = Image.open(img_dir)        # 打开img_dir路径下的图片
    image = image.resize([208, 208])   # 改变图片的大小，定为宽高都为208像素
    image = np.array(image)            # 转成多维数组，向量的格式
    return image


def evaluate_one_image():
    # 修改成自己测试集的文件夹路径
    test_dir = 'D:/WorkSpace/work_to_pycharm/cats_vs_dogs/data/test/'
    # test_dir = '/home/user/Dataset/cats_vs_dogs/test/'

    test_img = input_data.get_files(test_dir)[0]   # 获取测试集的图片路径列表
    image_array = get_one_image(test_img)          # 从测试集中随机选取一张图片

    # 将这个图设置为默认图，会话设置成默认对话，这样在with语句外面也能使用这个会话执行。
    with tf.Graph().as_default():    # 参考：https://blog.csdn.net/nanhuaibeian/article/details/101862790
        BATCH_SIZE = 1               # 这里我们要输入的是一张图(预测这张随机图)
        N_CLASSES = 2                # 还是二分类(猫或狗)

        image = tf.cast(image_array, tf.float32)                    # 将列表转换成tf能够识别的格式
        image = tf.image.per_image_standardization(image)           # 图片标准化处理
        image = tf.reshape(image, [1, 208, 208, 3])                 # 改变图片的形状
        logit = model.cnn_inference(image, BATCH_SIZE, N_CLASSES)   # 得到神经网络输出层的预测结果
        logit = tf.nn.softmax(logit)                                # 进行归一化处理（使得预测概率之和为1）

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])         # x变量用于占位，输入的数据要满足这里定的shape

        # 修改成自己训练好的模型路径
        logs_train_dir = 'D:/WorkSpace/work_to_pycharm/cats_vs_dogs/log/'

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("从指定路径中加载模型...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)   # 读取路径下的checkpoint
            # 载入模型，不需要提供模型的名字，会通过 checkpoint 文件定位到最新保存的模型
            if ckpt and ckpt.model_checkpoint_path:                # checkpoint存在且其存放的变量不为空
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]   # 通过切割获取ckpt变量中的步长
                saver.restore(sess, ckpt.model_checkpoint_path)    # 当前会话中，恢复该路径下模型的所有参数（即调用训练好的模型）
                print('模型加载成功, 训练的步数为： %s' % global_step)
            else:
                print('模型加载失败，checkpoint文件没找到！')

            # 通过saver.restore()恢复了训练模型的参数（即：神经网络中的权重值），这样logit才能得到想要的预测结果
            # 执行sess.run()才能运行，并返回结果数据
            prediction = sess.run(logit, feed_dict={x: image_array})   # 输入随机抽取的那张图片数据，得到预测值
            max_index = np.argmax(prediction)                          # 获取输出结果中最大概率的索引(下标)
            if max_index == 0:
                pre = prediction[:, 0][0] * 100
                print('图片是猫的概率为： {:.2f}%'.format(pre))       # 下标为0，则为猫，并打印是猫的概率
            else:
                pre = prediction[:, 1][0] * 100
                print('图片是狗的概率为： {:.2f}%'.format(pre))       # 下标为1，则为狗，并打印是狗的概率

    plt.imshow(image_array)                                        # 接受图片并处理
    plt.show()                                                     # 显示图片


if __name__ == '__main__':
    # 调用方法，开始测试
    evaluate_one_image()