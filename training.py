"""
    training.py: 模型的训练及评估
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
import matplotlib.pyplot as plt
import input_data
import model


N_CLASSES = 2  # 分类数，猫和狗
IMG_W = 208  # resize图片宽高，太大的话训练时间久
IMG_H = 208
BATCH_SIZE = 16  # 每批次读取数据的数量
CAPACITY = 2000  # 队列最大容量
MAX_STEP = 10000  # 训练最大步数，一般5K~10k
learning_rate = 0.0001  # 学习率，一般小于0.0001

# train_dir = 'D:/WorkSpace/Dataset/cats_vs_dogs/data/train/'        # 训练集的文件夹路径
# logs_train_dir = 'D:/WorkSpace/work_to_pycharm/cats_vs_dogs/log/'  # 记录训练过程与保存模型的路径
train_dir = '/home/user/Dataset/cats_vs_dogs/train/'                 # 训练集的文件夹路径
logs_train_dir = '/home/dujunjie/PycharmProjects/cats_vs_dogs/log/'  # 记录训练过程与保存模型的路径

# 获取要训练的图片和对应的图片标签, 这里返回的train_img是存放猫狗图片路径的列表，train_label是存放对train对应标签的列表(0是猫，1是狗)
train_img, train_label = input_data.get_files(train_dir)

# 读取队列中的数据
train_batch, train_label_batch = input_data.get_batch(train_img, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# 调用model方法得到返回值, 进行变量赋值
train_logits = model.cnn_inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)
train_op = model.training(train_loss, learning_rate)
train_acc = model.evaluation(train_logits, train_label_batch)

summary_op = tf.summary.merge_all()  # 将所有summary全部保存到磁盘，以便tensorboard显示

accuracy_list = []   # 记录准确率(每50步存一次)
loss_list = []       # 记录损失值(每50步存一次)
step_list = []       # 记录训练步数(每50步存一次)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())                        # 变量初始化，如果存在变量则是必不可少的操作

    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)   # 用于向logs_train_dir写入summary(训练)的目标文件
    saver = tf.train.Saver()                                           # 用于存储训练好的模型

    # 队列监控（训练的batch数据用到了队列）
    coord = tf.train.Coordinator()   # 创建线程协调器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # 执行MAX_STEP步的训练，一步一个batch
        for step in np.arange(MAX_STEP):
            if coord.should_stop():   # 队列中的所有数据已被读出，无数据可读时终止训练
                break

            _op, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])   # 在会话中才能读取tensorflow的变量值

            # 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_train = sess.run(summary_op)            # 调用sess.run()，生成的训练数据
                train_writer.add_summary(summary_train, step)   # 将训练过程及训练步数保存

            # 每隔100步画图，记录训练的准确率和损失值的结点
            if step % 100 == 0:
                accuracy_list.append(tra_acc)
                loss_list.append(tra_loss)
                step_list.append(step)

            # 每隔5000步，保存一次训练好的模型（即：训练好的模型的参数保存下来）
            if step % 5000 == 0 or (step + 1) == MAX_STEP:
                # ckpt文件是一个二进制文件，它把变量名映射到对应的tensor值
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        plt.figure()                    # 建立可视化图像框
        plt.plot(step_list, accuracy_list, color='b', label='cnn_accuracy')               # 蓝线为准确率
        plt.plot(step_list, loss_list, color='r', label='cnn_loss', linestyle='dashed')   # 红虚线为损失值
        plt.xlabel("Step")              # x轴取名
        plt.ylabel("Accuracy/Loss")     # y轴取名
        plt.legend()                    # 给图加上图例
        plt.show()                      # 显示图片

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()   # 停止所有线程

    coord.join(threads)   # 等待所有线程结束
    sess.close()          # 关闭会话
