import tensorflow as tf
import BasicFunction as bf


class ResNeXT50_FPN :


    def __init__(self,Graph,dataPH,LabelPH,trainingPH,learningRatePH):
        shapeOfInputData = dataPH.get_shape().as_list()
        #b,c,h,w
        assert len(shapeOfInputData) == 4
        self.__inputPH = dataPH
        self.__labelPH = LabelPH
        self.__trainingPH = trainingPH
        self.__learningRPH = learningRatePH
        self.__graph = Graph

    def __SmallUnitBlockBuild(self,inputTensor,i,j,outChannal):
        shapeOfInput = inputTensor.get_shape().as_list()
        with tf.variable_scope("Unite_Block" + str(i)+str(j)):
            with tf.variable_scope("1X1_Conv_First"):
                filterConv1 = bf.Weight(shape=[1,1,shapeOfInput[1],4],name="filterConv1",ifAddToRegul=True)
                convF = bf.Conv2d(inputTensor,filterConv1,strides=[1,1,1,1],name="ConvF")
                convF = bf.BatchNormalize(convF,training=self.__trainingPH,name="BNF")
                convF = tf.nn.leaky_relu(convF,alpha=0.001)
            with tf.variable_scope("3x3_Conv_Second"):
                filterConv2 = bf.Weight(shape=[3,3,4,4],name="filterConv2",ifAddToRegul=True)
                convS = bf.Conv2d(convF,filterConv2,strides=[1,1,1,1],name="ConvS")
                convS = bf.BatchNormalize(convS,training=self.__trainingPH,name="BNS")
                convS = tf.nn.leaky_relu(convS,alpha=0.001)
            with tf.variable_scope("1x1_Conv_Third"):
                filterConv3 = bf.Weight(shape=[1,1,4,outChannal],name="filterConv3",ifAddToRegul=True)
                convT = bf.Conv2d(convS,filterConv3,strides=[1,1,1,1],name="ConvT")
                convT = bf.BatchNormalize(convT,training=self.__trainingPH,name="BNT")
                convT = tf.nn.leaky_relu(convT,alpha=0.001)
        return convT

    def __OneUnitBulid(self,inputTensor,i,outChannal):
        shapeOfInput = inputTensor.get_shape().as_list()
        with tf.variable_scope("OneUniteBlock" + str(i)):
            inputTensor_copy = tf.identity(inputTensor)
            temporaryList = []
            for j in range(4):
                outputOneSmallUnit = self.__SmallUnitBlockBuild(inputTensor,i , j,outChannal)
                temporaryList.append(outputOneSmallUnit)
            addCollection = tf.add_n(temporaryList)
            shapeOfAddCollection = addCollection.get_shape().as_list()
            with tf.variable_scope("DoTransCopy"):
                filterW = bf.Weight(shape=[1,1,shapeOfInput[1],shapeOfAddCollection[1]]
                                    ,name="filterW",ifAddToRegul=True)
                copyTrans = bf.Conv2d(inputTensor_copy,filterW,strides=[1,1,1,1],name="TransCopy")
            out = tf.add(addCollection,copyTrans)
            out = tf.nn.leaky_relu(out,alpha=0.001)
        return out

    #tf.summary.FileWriter(logdir="logdir",graph=tf.get_default_graph())
    #return 2,3,4,5 layers
    def __BackBoneBuild(self):
        shapeOfInput = self.__inputPH.get_shape().as_list()
        with tf.variable_scope("Conv1_7X7"):
            filterConv1 = bf.Weight(shape=[7,7,shapeOfInput[1],64],name="filterConv1W",ifAddToRegul=True)
            Conv1 = bf.Conv2d(self.__inputPH,fil=filterConv1,strides=[1,1,2,2],name="Conv1_2d")
            Conv1 = bf.BatchNormalize(Conv1,training=self.__trainingPH,name="BN_Conv1")
            Conv1 = tf.nn.leaky_relu(Conv1,alpha=0.001)
        print("shape of conv1 is ", Conv1.get_shape().as_list())
        #Max pool with 3*3 stride 2
        with tf.variable_scope("Conv1_7X7_MaxPool"):
            Conv2 = bf.Pool(Conv1,windowShape=[3,3],strides=[2,2],ptype="MAX",name="MaxPoolWithConv1")
        #First layers
        with tf.variable_scope("First_layers"):
            numberOfFirstLayers = 3
            for i in range(numberOfFirstLayers):
                Conv2 = self.__OneUnitBulid(Conv2,i,outChannal=256)
        print("shape of conv2 is ", Conv2.get_shape().as_list())
        Conv2_Copy = tf.identity(Conv2)
        #Second layers
        with tf.variable_scope("Conv2_3x3_MaxPool"):
            Conv3 = bf.Pool(Conv2,windowShape=[3,3],strides=[2,2],ptype="MAX",name="MaxPoolWithConv2")
        with tf.variable_scope("Second_layers"):
            numberOfSecondLayers = 4
            for i in range(numberOfSecondLayers):
                Conv3 = self.__OneUnitBulid(Conv3,i,outChannal=512)
        print("shape of conv3 is ", Conv3.get_shape().as_list())
        Conv3_Copy = tf.identity(Conv3)
        #Third layers
        with tf.variable_scope("Conv3_3x3_MaxPool"):
            Conv4 = bf.Pool(Conv3,windowShape=[3,3],strides=[2,2],ptype="MAX",name="MaxPoolWithConv3")
        with tf.variable_scope("Third_Layers"):
            numberOfThirdLayers = 6
            for i in range(numberOfThirdLayers):
                Conv4 = self.__OneUnitBulid(Conv4,i,outChannal=1024)
        print("shape of conv4 is ", Conv4.get_shape().as_list())
        Conv4_Copy = tf.identity(Conv4)
        #Fourth layers
        with tf.variable_scope("Conv4_3x3_MaxPool"):
            Conv5 = bf.Pool(Conv4,windowShape=[3,3],strides=[2,2],ptype="MAX",name="MaxPoolWithConv4")
        with tf.variable_scope("Fourth_Layers"):
            numberOfFourthLayers = 3
            for i in range(numberOfFourthLayers):
                Conv5 = self.__OneUnitBulid(Conv5,i,outChannal=2048)
        print("shape of conv5 is ", Conv5.get_shape().as_list())
        Conv5_Copy = tf.identity(Conv5)
        #globel average pool
        shapeOfConv5 = Conv5.get_shape().as_list()
        with tf.variable_scope("Avg_Pooling"):
            FCNodes = bf.Pool(Conv5,windowShape=[shapeOfConv5[2],shapeOfConv5[3]],strides=[1,1],ptype="AVG",
                              name="AVG_POOL",padding="VALID")
        shapeOfFCNodes = FCNodes.get_shape().as_list()
        nodesNumbers = int(shapeOfFCNodes[1] * shapeOfFCNodes[2] * shapeOfFCNodes[3])
        FCNodes = tf.reshape(FCNodes,shape=[-1,nodesNumbers])
        with tf.variable_scope("Full_Connection"):
            Weight1 = bf.Weight(shape = [nodesNumbers,1000],ifAddToRegul=True,name="Weight1")
            bias1 = tf.constant(value=0.,shape=[1000],dtype=tf.float32,name="bias1")
            FullC1 = tf.add(tf.matmul(FCNodes,Weight1),bias1)
            FullC1 = tf.nn.leaky_relu(FullC1,alpha=0.001)
            Weight2 = bf.Weight(shape=[1000,256],ifAddToRegul=True,name="Weight2")
            bias2 = tf.constant(value=0., shape=[256], dtype=tf.float32, name="bias2")
            FullC2 = tf.add(tf.matmul(FullC1,Weight2),bias2)
            FullC2 = tf.nn.leaky_relu(FullC2,alpha=0.001)
        return Conv2_Copy,Conv3_Copy,Conv4_Copy,Conv5_Copy,FullC2

    @staticmethod
    def __UpSampleing(inputTensor,filterWeight,outputShape):
        return tf.nn.conv2d_transpose(inputTensor,filterWeight,output_shape=outputShape,padding="SAME",
                                      strides=[1,1,2,2],data_format="NCHW",name = "UpSample")

    def NetBuild(self):
        with self.__graph.as_default():
            Conv2_Copy, Conv3_Copy, Conv4_Copy, Conv5_Copy, FCOri = self.__BackBoneBuild()
            tf.add_to_collection("FC", FCOri)
            shapeOfConv5Copy = Conv5_Copy.get_shape().as_list()
            shapeOfConv4Copy = Conv4_Copy.get_shape().as_list()
            shapeOfConv3Copy = Conv3_Copy.get_shape().as_list()
            shapeOfConv2Copy = Conv2_Copy.get_shape().as_list()
            #####

            with tf.variable_scope("1x1_Conv5_Copy"):
                filterP5 = bf.Weight(shape=[1, 1, shapeOfConv5Copy[1], 256], name="P5Filter", ifAddToRegul=True)
                P5 = bf.Conv2d(Conv5_Copy, filterP5, strides=[1, 1, 1, 1], name="P5Conv")
            with tf.variable_scope("1x1_Conv4_Copy"):
                filterP4 = bf.Weight(shape=[1, 1, shapeOfConv4Copy[1], 256], name="P4Filter", ifAddToRegul=True)
                P4 = bf.Conv2d(Conv4_Copy, filterP4, strides=[1, 1, 1, 1], name="P4Conv")
            with tf.variable_scope("1x1_Conv3_Copy"):
                filterP3 = bf.Weight(shape=[1, 1, shapeOfConv3Copy[1], 256], name="P3Filter", ifAddToRegul=True)
                P3 = bf.Conv2d(Conv3_Copy, filterP3, strides=[1, 1, 1, 1], name="P3Conv")
            with tf.variable_scope("1x1_Conv2_Copy"):
                filterP2 = bf.Weight(shape=[1, 1, shapeOfConv2Copy[1], 256], name="P2Filter", ifAddToRegul=True)
                P2 = bf.Conv2d(Conv2_Copy, filterP2, strides=[1, 1, 1, 1], name="P2Conv")
            ####

            with tf.variable_scope("P5_3x3"):
                filterP5_3X3 = bf.Weight(shape=[3, 3, 256, 256], name="P5_3x3_w", ifAddToRegul=True)
                P5_3x3 = bf.Conv2d(P5, filterP5_3X3, strides=[1, 1, 1, 1], name="3x3_P5")
                print("shape of P5 is ", P5_3x3.get_shape().as_list())
            with tf.variable_scope("P4_3X3"):
                shapeOfP4 = P4.get_shape().as_list()
                filterUpSampleP5 = bf.Weight(shape=[3, 3, 256, 256], name="UpP5", ifAddToRegul=True)
                P5UP = self.__UpSampleing(P5, filterUpSampleP5, [shapeOfP4[0], shapeOfP4[1], shapeOfP4[2], shapeOfP4[3]])
                P5_P4_Add = tf.add(P4, P5UP)
                filterP4_3X3 = bf.Weight(shape=[3, 3, 256, 256], name="P4_3x3_w", ifAddToRegul=True)
                P4_3x3 = bf.Conv2d(P5_P4_Add, filterP4_3X3, strides=[1, 1, 1, 1], name="3x3_p4")
                print("shape of P4 is ", P4_3x3.get_shape().as_list())
            with tf.variable_scope("P3_3x3"):
                shapeOfP3 = P3.get_shape().as_list()
                filterUpSampleP4 = bf.Weight(shape=[3, 3, 256, 256], name="UpP4", ifAddToRegul=True)
                P4UP = self.__UpSampleing(P4, filterUpSampleP4, [shapeOfP3[0], shapeOfP3[1], shapeOfP3[2], shapeOfP3[3]])
                P4_P3_Add = tf.add(P4UP, P3)
                filterP3_3X3 = bf.Weight(shape=[3, 3, 256, 256], name="P3_3x3_w", ifAddToRegul=True)
                P3_3x3 = bf.Conv2d(P4_P3_Add, filterP3_3X3, strides=[1, 1, 1, 1], name="3x3_p3")
                print("shape of P3 is ", P3_3x3.get_shape().as_list())
            with tf.variable_scope("P2_3X3"):
                shapeOfP2 = P2.get_shape().as_list()
                filterUpSampleP3 = bf.Weight(shape=[3, 3, 256, 256], name="UpP3", ifAddToRegul=True)
                P3UP = self.__UpSampleing(P3, filterUpSampleP3, [shapeOfP2[0], shapeOfP2[1], shapeOfP2[2], shapeOfP2[3]])
                P3_P2_Add = tf.add(P3UP, P2)
                filterP2_3X3 = bf.Weight(shape=[3, 3, 256, 256], name="P2_3x3_w", ifAddToRegul=True)
                P2_3x3 = bf.Conv2d(P3_P2_Add, filterP2_3X3, strides=[1, 1, 1, 1], name="3x3_p2")
                print("shape of P2 is ", P2_3x3.get_shape().as_list())
            ####

            with tf.variable_scope("AVG_POOL_P5"):
                shapeOfP5_3x3 = P5_3x3.get_shape().as_list()
                FCNodesP5 = bf.Pool(P5_3x3, windowShape=[shapeOfP5_3x3[2], shapeOfP5_3x3[3]], strides=[1, 1]
                                    , ptype="AVG", padding="VALID")
                FCP5 = tf.reshape(FCNodesP5, shape=[-1, 256])
                tf.add_to_collection("FC", FCP5)
            with tf.variable_scope("AVG_POOL_P4"):
                shapeOfP4_3x3 = P4_3x3.get_shape().as_list()
                FCNodesP4 = bf.Pool(P4_3x3, windowShape=[shapeOfP4_3x3[2], shapeOfP4_3x3[3]],
                                    strides=[1, 1], ptype="AVG", padding="VALID")
                FCP4 = tf.reshape(FCNodesP4, shape=[-1, 256])
                tf.add_to_collection("FC", FCP4)
            with tf.variable_scope("AVG_POOL_P3"):
                shapeOfP3_3x3 = P3_3x3.get_shape().as_list()
                FCNodesP3 = bf.Pool(P3_3x3, windowShape=[shapeOfP3_3x3[2], shapeOfP3_3x3[3]],
                                    strides=[1, 1], ptype="AVG", padding="VALID")
                FCP3 = tf.reshape(FCNodesP3, shape=[-1, 256])
                tf.add_to_collection("FC", FCP3)
            with tf.variable_scope("AVG_POOL_P2"):
                shapeOfP2_3x3 = P2_3x3.get_shape().as_list()
                FCNodesP2 = bf.Pool(P2_3x3, windowShape=[shapeOfP2_3x3[2], shapeOfP2_3x3[3]],
                                    strides=[1, 1], ptype="AVG", padding="VALID")
                FCP2 = tf.reshape(FCNodesP2, shape=[-1, 256])
                tf.add_to_collection("FC", FCP2)
            shapeOfLabel = self.__labelPH.get_shape().as_list()

            with tf.variable_scope("FULL_Connection_All"):
                finalFC = tf.add_n(tf.get_collection("FC"))
                finalFC = bf.BatchNormalize(finalFC, training=self.__trainingPH, name="FCBNTrans")
                WeightF = bf.Weight(shape=[256, shapeOfLabel[1]], name="WeightF", ifAddToRegul=True)
                BiasF = tf.constant(0., dtype=tf.float32, shape=[shapeOfLabel[1]], name="BiasF")
                beforeTanh = tf.add(BiasF, tf.matmul(finalFC, WeightF))
                netOutput = tf.nn.tanh(beforeTanh)
        return beforeTanh , netOutput


    def LossAndOptimizerBuild(self,netOut):
        with self.__graph.as_default():
            Loss = tf.reduce_mean(tf.squared_difference(netOut, self.__labelPH, name="LabelLoss"))
            tf.add_to_collection("Loss", Loss)
            tLoss = tf.add_n(tf.get_collection("Loss"))
            UPDATes = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print("Start build Optimizer , Please wait for a moment .")
            with tf.control_dependencies(UPDATes):
                optim = tf.train.MomentumOptimizer(learning_rate=self.__learningRPH, momentum=0.9,
                                                   use_nesterov=True).minimize(tLoss)
            print("All build has been completed .")
        return tLoss,optim


if __name__ == "__main__":
    batch_size = 16
    graph = tf.get_default_graph()
    inputPlaceHolder = tf.placeholder(shape=[None,3,256,256],dtype=tf.float32)
    labelPlaceHolder = tf.placeholder(shape=[None,1],dtype=tf.float32)
    trainingPlaceHloder = tf.placeholder(dtype=tf.bool)
    lrPH = tf.placeholder(dtype=tf.float32)
    XT101 = ResNeXT50_FPN(graph,inputPlaceHolder,labelPlaceHolder,trainingPlaceHloder,lrPH)
    beOut , netResult= XT101.NetBuild()
    totalLoss , Optimi = XT101.LossAndOptimizerBuild(netResult)



