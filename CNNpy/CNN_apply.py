#-*- coding:utf-8 -*-
import tensorflow as tf
from PIL import Image
import numpy as np
import CNN_inference
import CNN_train

IMAGE_PATH = "dataSets/handWriteData/s1/one.jpeg"

def restore_model(PicArr):
    with tf.Graph().as_default() as tg:
        #reshaped_Pic_Arr = np.reshape(PicArr, [1, CNN_inference.IMAGE_SIZE, CNN_inference.IMAGE_SIZE, CNN_inference.NUM_CHANNELS])
        x = tf.placeholder(tf.float32, [1, CNN_inference.IMAGE_SIZE, CNN_inference.IMAGE_SIZE, CNN_inference.NUM_CHANNELS])
        y = CNN_inference.inference(x ,None, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(CNN_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(CNN_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue = sess.run(preValue, feed_dict={x: PicArr}) 
                return preValue
            else :
                print("NO MODEL FOUND!")
                return -1

def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28,28), Image.ANTIALIAS) #用消除锯齿的方法resize
    im_arr = np.array(reIm.convert('L'))
    #阈值为50 小于50认为是黑色点，否则是白点，下面的循环给图片反色并过滤噪声
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if(im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    
    nm_arr = im_arr.reshape([1, 28, 28, 1])
    nm_arr = nm_arr.astype(np.float32)
    #从0-255变为0-1
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)
    return img_ready

def application(imgPath):
    testPicArr = pre_pic(imgPath)
    predictionValue = restore_model(testPicArr)
    print("#########################Prediction Value :%d ####################" % predictionValue)

if __name__ == "__main__":
    application(IMAGE_PATH)

