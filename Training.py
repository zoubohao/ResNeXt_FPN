import tensorflow as tf
import numpy as np
import cv2
import skimage.io as io
import random
import math
from ResNeXT50_FPN import ResNeXT50_FPN
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


path_of_training_file = "c:\\train_DeOrNot_info.txt"
path_of_val_file = "c:\\val_DeOrNot_info.txt"
path_of_weight_file = "E:\\ResNeXt_FPN_weight\\2\\ResNeXt_FPN.ckpt"
path_list_of_images = ["e:\\train_all\\",
                       "e:\\train_allHistMeanTrans\\",
                       "e:\\train_reorganizedImage\\",
                       "e:\\train_reorganizedHistMeanTrans\\"]
loadWeights = "yes"
command = "test"
displaySteps = 25
iniLearningRate = 5e-6
decayRate = 0.96
decayStep = 5000
epoch = 20
timesInOneEpoch = 5000
batch_size = 4
height = 512
width = 512
channal = 1
labelNumber = 1


#######################
#build dictionary with images and labels
dicTrainingLungOpacity = {}
dicTrainingOthers = {}
dicValid = {}
print("Reading data.")
with open(path_of_training_file,mode="r") as f :
    for oneline in f:
        items = oneline.strip("\n").split(",")
        if items[1] == "Lung Opacity":
            dicTrainingLungOpacity[items[0] + ".jpg"] = items[1]
        if items[1] == "No Lung Opacity / Not Normal" or items[1] == "Normal":
            dicTrainingOthers[items[0] + ".jpg"] = items[1]

with open(path_of_val_file , mode="r") as f :
    for oneline in f :
        items = oneline.strip("\n").split(",")
        dicValid[items[0] + ".jpg"] = items[1]
print("Reading has completed .")
print("LungOpacity samples are ",len(dicTrainingLungOpacity))
print("Others samples are ",len(dicTrainingOthers))
print("Valid samples are ",len(dicValid))

def generateData (listOfData):
    while True:
        for ele in listOfData:
            yield ele

#########################
#place holder build
graph = tf.Graph()
with graph.as_default():
    inputPlaceHolder = tf.placeholder(shape=[batch_size, channal, height, width], dtype=tf.float32)
    labelPlaceHolder = tf.placeholder(shape=[batch_size, labelNumber], dtype=tf.float32)
    trainingPH = tf.placeholder(tf.bool)
    learningPH = tf.placeholder(tf.float32)

##########################
#Net build
XT_FPN = ResNeXT50_FPN(graph,inputPlaceHolder,labelPlaceHolder,trainingPH,learningPH)
beOut , netOut = XT_FPN.NetBuild()
tLoss , optimi = XT_FPN.LossAndOptimizerBuild(netOut)
#tf.summary.FileWriter(logdir="logdir",graph=graph)

##########################
#build session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
numberOfDataSource = len(path_list_of_images)
with tf.Session(graph=graph,config=config) as sess:
    if loadWeights.lower() == "yes":
        tf.train.Saver().restore(sess, path_of_weight_file)
    else:
        print("Start initial variables .")
        sess.run(tf.global_variables_initializer())
    if command.lower() == "train":
        print("initial variable has been done. ")
        listOfLungOpacity = list(dicTrainingLungOpacity.keys())
        listOfOthers = list(dicTrainingOthers.keys())
        geneLung = generateData(listOfLungOpacity)
        geneOther = generateData(listOfOthers)
        learningR = iniLearningRate
        totalTrainingTimes = 0
        for e in range(epoch):
            for t in range(timesInOneEpoch):
                dataImages = []
                dataLabels = []
                if totalTrainingTimes % decayStep == 0 and totalTrainingTimes != 0:
                    learningR = learningR * math.pow(decayRate * 1.0, (totalTrainingTimes * 1.0 / decayStep))
                for b in range(int(batch_size / 2.0)):
                    randomNumber = int(random.random() * numberOfDataSource + 0.)
                    imageID = geneLung.__next__()
                    path_of_image = path_list_of_images[randomNumber] + imageID
                    image = io.imread(path_of_image,as_gray=True)
                    if image.shape[0] != height or image.shape[1] != width:
                        image = cv2.resize(image,(height,width),interpolation=cv2.INTER_CUBIC)
                    image = image / 255.
                    image = np.reshape(image,newshape=[1,height,width])
                    dataImages.append(image)
                    dataLabels.append([1.])
                for b in range(int(batch_size / 2.0)):
                    randomNumber = int(random.random() * numberOfDataSource + 0.)
                    imageID = geneOther.__next__()
                    path_of_image = path_list_of_images[randomNumber] + imageID
                    image = io.imread(path_of_image,as_gray=True)
                    if image.shape[0] != height or image.shape[1] != width:
                        image = cv2.resize(image,(height,width),interpolation=cv2.INTER_CUBIC)
                    image = image / 255.
                    image = np.reshape(image,newshape=[1,height,width])
                    dataImages.append(image)
                    dataLabels.append([-1.])
                dataImages = np.array(dataImages)
                dataLabels = np.array(dataLabels)
                if totalTrainingTimes % displaySteps == 0:
                    print("Labels are ",dataLabels)
                    beOutRes = sess.run(beOut,feed_dict={
                        inputPlaceHolder:dataImages,
                        trainingPH : False
                    })
                    print("Before output is ",beOutRes)
                    netOutRes = sess.run(netOut,feed_dict={
                        inputPlaceHolder: dataImages,
                        trainingPH: False
                    })
                    print("Net out is ",netOutRes)
                    tLossRes = sess.run(tLoss,feed_dict={
                        inputPlaceHolder: dataImages,
                        labelPlaceHolder: dataLabels,
                        trainingPH: False
                    })
                    print("total loss is ",tLossRes)
                    print("Step is ",totalTrainingTimes)
                    print("Learning rate is ",learningR)
                sess.run(optimi,feed_dict={
                    inputPlaceHolder:dataImages,
                    labelPlaceHolder:dataLabels,
                    trainingPH:True,
                    learningPH:learningR
                })
                totalTrainingTimes += 1
            tf.train.Saver().save(sess,save_path=path_of_weight_file)
    else:
        TH = 0
        TP = 0
        TP_FN = 0
        TN = 0
        TN_FP = 0
        ACC = 0
        TOT = 0
        for key , value in dicValid.items():
            randomNumber = int(random.random() * numberOfDataSource + 0.)
            path_of_image = path_list_of_images[randomNumber] + key + ".jpg"
            image = io.imread(path_of_image , as_gray = True)
            if image.shape[0] != height or image.shape[1]  != width :
                image = cv2.resize(image,(height,width),interpolation= cv2.INTER_CUBIC)
            image = image / 255.
            print("ID is ",key)
            print("value is ",value)
            calNetOut = sess.run(netOut,feed_dict={
                inputPlaceHolder : image,
                trainingPH : False
            })
            print("Net calculation result is ",calNetOut)
            TOT += 1
            if value == "Lung Opacity":
                TP_FN += 1
                if calNetOut > TH :
                    TP += 1
                    ACC += 1
            if value == "No Lung Opacity / Not Normal" or value == "Normal":
                TN_FP += 1
                if calNetOut < TH:
                    TN += 1
                    ACC += 1
        print("TP ",TP)
        print("TP_FN ",TP_FN)
        print("TN ",TN)
        print("TN_FP ",TN_FP)
        print("TP / TP_FN ",TP / TP_FN + 0.0)
        print("TN / TN_FP ",TN / TN_FP + 0.0)
        print("ACC / TOT ",ACC / TOT + 0.0 )







