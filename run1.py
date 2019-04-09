import tensorflow as tf
import numpy as np
import cv2
from skimage import io, transform
import os
from PIL import Image

# 定义读取图片信息的函数
def inputimage(input_img_path, file):
    img = io.imread(input_img_path + file)  # 用skimage读取目标图片
    output_height = img.shape[0]  # 得到目标图片的长和宽
    output_width = img.shape[1]
    inputimg = transform.resize(img, (416, 416, 3))  # 将图片尺寸改成yolo输入的图片尺寸
    inputimg = np.reshape(inputimg, [-1, 416, 416, 3])# 使用numpy把图片转换成矩阵
    return output_height, output_width, inputimg

# 定义展示结果的函数
def showresult(input_img_path, file, predictionsdir, logo):
    im = Image.open(predictionsdir)  # 用PIL贴商标
    im.paste(logo, (0, 0))  # 贴到左上角(0,0)的位置
    im.save(predictionsdir)  # 保存

    predictions = cv2.imread(predictionsdir)  # 读取处理好的结果图
    oimage = cv2.imread(input_img_path + file)  # 读取原图片
    cv2.imshow("Predictions", predictions)  # 展示结果图
    cv2.imshow("Original", oimage)  # 展示原图
    cv2.waitKey(1500)  # 等待3秒，处理下一张图片

# 定义sigmoid激活函数，用以给置信度加入非线性因素
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# 定义softmax函数，给数组中的每个数值分配权重，数值大的会分配更大的比重
def softmax(x):
    e_x = np.exp(x - np.max(x))# 处理对象是数组
    out = e_x / e_x.sum()
    return out

# 定义iou函数，计算识别网格和定义网格的重合度
def iou(boxA, boxB):

    # 确定两个矩形交集的对角坐标（确定两个对角的坐标，即可确定面积）
    xA = max(boxA[0], boxB[0])#左上角横坐标
    yA = max(boxA[1], boxB[1])#左上角纵坐标
    xB = min(boxA[2], boxB[2])#右下角横坐标
    yB = min(boxA[3], boxB[3])#右下角纵坐标

    # 计算交集矩形的总面积
    intersection_area = (xB - xA + 1) * (yB - yA + 1)

    # 分别计算两个矩形的面积
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # 计算两个矩形的重合程度，重合度 = 交集面积 / ( 矩形A面积 + 矩形B面积 - 交集面积 )
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou # 函数返回重合度

# 定义函数，在置信度合格的目标框中，将区域重合的多个只保留一个
def non_maximal_suppression(thresholded_predictions, iou_threshold):
    nms_predictions = []# 定义数组，包存本函数的输出结果

    # 首先将置信度最高的目标框添加到输出数组中，保证置信度最高的网格不会被删除掉
    nms_predictions.append(thresholded_predictions[0])

    # 从第二个目标框开始，剔除高度重合的目标框
    # thresholded_predictions[i][0] = [x1,y1,x2,y2]
    # thresholded_predictions每行的第一个元素是目标框的左上和右下坐标
    i = 1
    while i < len(thresholded_predictions):
        n_boxes_to_check = len(nms_predictions)# 得出置信度更高，并且已经验证与其他目标框重合低的目标框数量
        to_delete = False# 删除该目标框的标志位

        j = 0
        while j < n_boxes_to_check:
            # 遍历已经确认输出的目标框，计算本目标框与每个框的重合度
            curr_iou = iou(thresholded_predictions[i][0], nms_predictions[j][0])
            # 如果被验证的框，与任何一个置信度更高的框重合度高，将删除标志位改成TRUE
            if (curr_iou > iou_threshold):
                to_delete = True
            j = j + 1

        # 如果删除标志位是False，将该框加入到输出中
        if to_delete == False:
            nms_predictions.append(thresholded_predictions[i])
        i = i + 1

    return nms_predictions

# 处理网络计算结果的函数
def postprocessing(predictions, input_img_path, score_threshold, iou_threshold, output_height, output_width):
    input_image = cv2.imread(input_img_path)# 读取图片
    input_image = cv2.resize(input_image, (output_width, output_height), interpolation=cv2.INTER_CUBIC)# 将图片拉伸成网络的输出尺寸

    n_grid_cells = 13# 整个图片被分成13*13个区域
    n_b_boxes = 5#每个区域定义5个box，可以理解成用以画框匹配目标框

    # 定义标签与RGB颜色
    classes = ["yuyin", "baojing", "tft", "daozha", "xianshi", "ludeng", "jingtai", "chongdian", "jiaotong", "cheku"]
    colors = [(254.0, 0, 254), (239.88888888888889, 211.66666666666669, 127),
              (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
              (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
              (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
              (141.11111111111111, -84.66666666666664, 0), (127.0, 0, 254)]

    # tiny-yolo官方定义的anchors数值，在13*13个区域，每个区域根据anchors有5个B-Boxes
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    thresholded_predictions = []# 定义数组，置信度合格的网格数据将被添加到该数组中
    print('Threshold = {}'.format(score_threshold))

    # 将神经网络计算得到的矩阵重新变形 矩阵的参数 = [ 13 x 13 x (5个B-Boxes) x (4个边框位置值 + 1个目标分数 + 10个目标分类概率) ]
    # yolo网络输出包括13x13个区域（grid cells）的参数, 每个区域有5个目标框（B-Boxes）, 每个目标框有15个参数：4个边框位置（中心坐标+横纵距离）, 1个目标分数（包含目标中心的概率） , 10种分类概率（该目标属于哪一类）
    predictions = np.reshape(predictions, (13, 13, 5, 15))

    # 遍历13*13个目标区域，挑选出置信度高于threshold的目标框
    for row in range(n_grid_cells):
        for col in range(n_grid_cells):
            for b in range(n_b_boxes):# 遍历每个目标区域中的5个目标框

                tx, ty, tw, th, tc = predictions[row, col, b, :5]# 每个区域中的钱5个值代表边框位置和目标分数

                # 每个区域的长和宽都是32像素
                # YOLOv2预测必须转换为区域全尺寸的参数化坐标
                # 该计算方法是官方定义的
                center_x = (float(col) + sigmoid(tx)) * 32.0# 目标中心位置转换成图上x轴实际横坐标
                center_y = (float(row) + sigmoid(ty)) * 32.0# 目标中心位置转换成图上y轴实际横坐标
                roi_w = np.exp(tw) * anchors[2 * b + 0] * 32.0# 计算目标框距离中心的横轴距离
                roi_h = np.exp(th) * anchors[2 * b + 1] * 32.0# 计算目标框距离中心的纵轴距离
                # 计算目标框的四角位置，中心位置加减距离即可得到。
                left = int(center_x - (roi_w / 2.))
                right = int(center_x + (roi_w / 2.))
                top = int(center_y - (roi_h / 2.))
                bottom = int(center_y + (roi_h / 2.))

                final_confidence = sigmoid(tc)# 使用sigmoid激活函数计算该目标框包含目标中心的概率

                # 找到该目标框中目标的最优分类
                class_predictions = predictions[row, col, b, 5:]# 取出每个目标框的后10个参数，代表10种分类的概率
                class_predictions = softmax(class_predictions)# 使用softmax函数给分类概率重新分配权重
                class_predictions = tuple(class_predictions)# 把概率数组转换成元组，便于查找
                best_class = class_predictions.index(max(class_predictions))# 找到概率最大值所在的位置，即最优分类
                best_class_score = class_predictions[best_class]# 得到最大概率分类的概率

                # 置信度 = B-boxes包含目标中心的概率 * 最优分类的概率
                # 置信度高于主函数定义的threshold，即可通过第一次筛选
                if ((final_confidence * best_class_score) > score_threshold):
                    # 置信度通过后，记录目标框位置、置信度和最优分类
                    thresholded_predictions.append(
                        [[left, top, right, bottom], final_confidence * best_class_score, classes[best_class]])

    # 根据置信度将所有目标框（B-boxes）排序
    thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)#根据第2个元素排序，在第147行，该元素就是置信度

    if len(thresholded_predictions) == 0 :
        # 如果没有B-boxes通过置信度筛选，直接返回原图，不需要画框等操作
        return input_image

    print('These {} B-boxes has a score higher than threshold:'.format(len(thresholded_predictions)))
    for i in range(len(thresholded_predictions)):
        # 打印通过置信度筛选的目标框信息
        print('B-Box {} : {}'.format(i + 1, thresholded_predictions[i]))

    # 因为可能存在同一目标在多个目标框中的置信度很高，从而对同一目标识别出多个目标框
    # 所以要通过目标框的重合度来筛选，多个目标框彼此重合度比较高的情况下，只保留置信度最高的一个
    print('IOU higher than {} will be considered as the same object'.format(iou_threshold))
    nms_predictions = non_maximal_suppression(thresholded_predictions, iou_threshold)# 得到去除重复的目标框集合

    # 打印最终识别出的目标
    print('{} B-Boxes has the finial object:'.format(len(nms_predictions)))
    for i in range(len(nms_predictions)):
        print('B-Box {} : {}'.format(i + 1, nms_predictions[i]))

    # 为识别出的目标画框
    for i in range(len(nms_predictions)):
        color = colors[classes.index(nms_predictions[i][2])]# 每种分类的目标画框颜色不同
        best_class_name = nms_predictions[i][2]# 最优分类的标签名称
        score = str(nms_predictions[i][1]*100)[:4]# 得出正确率的前四位，保留到小数点后两位
        labels = best_class_name + " " + score +"%"# 组合出目标的标签和准确率，打印到图上
        # yolo网络使用416*416的图片预测，在不同尺寸的原图上画框，通过换算得知左上角和右下角的实际坐标
        start_x = int(nms_predictions[i][0][0]*output_width/416)
        start_y = int(nms_predictions[i][0][1]*output_height/416)
        end_x = int(nms_predictions[i][0][2]*output_width/416)
        end_y = int(nms_predictions[i][0][3]*output_height/416)

        # 画框并添加标签和置信度
        input_image = cv2.rectangle(input_image, (start_x, start_y), (end_x, end_y), color,5)# 用框画出目标框
        input_image = cv2.rectangle(input_image, (start_x-3, start_y), (start_x+len(labels)*14, start_y+20), color, -1)# 在目标上方，画实心矩形作为标签的背景
        cv2.putText(input_image, labels, (start_x,start_y+15),cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (255,255,255), 1)# 将标签的文字打印到图上

    return input_image # 返回处理后的图像

def main(_):
    # 定义各种路径
    input_img_path = '../bkrc_v1/'# 保存黑色背景测试图片的路径
    model = './model/bkrc_v1.pb'# pb格式的黑色背景测试模型
    # input_img_path = '../bkrc_v2/'# 保存彩色背景测试图片的路径
    # model = './model/bkrc_v2.pb'# pb格式的彩色背景测试模型
    logoimage = "./images/bkrclogo.jpg"# logo图片的保存地址
    predictionsdir = "./images/predictions.jpg"# YOLO输出结果图的缓存地址

    logo = Image.open(logoimage)# 读取logo图片


    # 定义两个参数，挑选概率高的目标识别。
    score_threshold = 0.3 # 筛选置信度，置信度 = 目标类别最大概率*该网格拥有目标的中心的概率
    iou_threshold = 0.3 # 筛选标记框的重合程度

    with tf.Graph().as_default():# 进入Tensorflow的默认图
        output_graph_def = tf.GraphDef()

        with open(model, "rb") as f:# 把pb格式模型文件读取到图中的默认会话里
            output_graph_def.ParseFromString(f.read())# 读取模型中的数据
            _ = tf.import_graph_def(output_graph_def, name="")# 配置到图中
       
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:# 进入默认会话
            init = tf.global_variables_initializer()# 初始化变量的语句
            sess.run(init)# 会话运行初始化变量

            #网络结构和参数已经定义好，定义变量来通过模型中的名字，代表网络的输入和输出
            input_x = sess.graph.get_tensor_by_name("input:0")# 定义模型中输入的变量（便于识别时feed用）
            out_label = sess.graph.get_tensor_by_name("output:0")# 定义程序里，代表模型中输处的变量

            while(1):
                for root, dirs, files in os.walk(input_img_path):# 遍历目标文件夹中的图片
                    for file in files:# 只遍历图片文件，不管路径和子文件夹

                        # 读取图片，得到原图的尺寸，并将原图转换成对应tiny_yolo网络输入的416*416*3矩阵
                        output_height, output_width, inputimg = inputimage(input_img_path, file)

                        print("Start Recognizing")
                        # 把图片的矩阵放入网络中识别
                        # 得到网络的输出，包括每个网格的目标概率、信任值和目标尺寸等数据
                        img_out = sess.run(out_label, feed_dict={input_x: inputimg})
                        print("Finish")

                        # 网络输出的信息和图片一起，放到函数中选出识别概率高的目标，并且画框标识
                        output_image = postprocessing(img_out, input_img_path+file, score_threshold, iou_threshold,
                                              output_height, output_width)
                        cv2.imwrite(predictionsdir, output_image)# 保存结果

                        #展示结果
                        showresult(input_img_path, file, predictionsdir, logo)




if __name__ == '__main__':
    tf.app.run(main=main)
