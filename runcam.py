import tensorflow as tf
import numpy as np
import cv2
from skimage import io, transform
import os




def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def iou(boxA, boxB):

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    intersection_area = (xB - xA + 1) * (yB - yA + 1)

    # Compute the area of both rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IOU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou


def non_maximal_suppression(thresholded_predictions, iou_threshold):
    nms_predictions = []
    print(thresholded_predictions)

    # Add the best B-Box because it will never be deleted
    nms_predictions.append(thresholded_predictions[0])

    # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
    # thresholded_predictions[i][0] = [x1,y1,x2,y2]
    i = 1
    while i < len(thresholded_predictions):
        n_boxes_to_check = len(nms_predictions)
        # print('N boxes to check = {}'.format(n_boxes_to_check))
        to_delete = False

        j = 0
        while j < n_boxes_to_check:
            curr_iou = iou(thresholded_predictions[i][0], nms_predictions[j][0])
            if (curr_iou > iou_threshold):
                to_delete = True
            # print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
            j = j + 1

        if to_delete == False:
            nms_predictions.append(thresholded_predictions[i])
        i = i + 1

    return nms_predictions


def preprocessing(input_img_path, input_height, input_width):
    input_image = cv2.imread(input_img_path)

    # Resize the image and convert to array of float32
    resized_image = cv2.resize(input_image, (input_height, input_width), interpolation=cv2.INTER_CUBIC)
    image_data = np.array(resized_image, dtype='f')

    # Normalization [0,255] -> [0,1]
    image_data /= 255.

    # Add the dimension relative to the batch size needed for the input placeholder "x"
    image_array = np.expand_dims(image_data, 0)  # Add batch dimension

    return image_array


def postprocessing(predictions, input_img, score_threshold, iou_threshold, output_height, output_width):
    input_image = input_img
    input_image = cv2.resize(input_image, (output_width, output_height), interpolation=cv2.INTER_CUBIC)

    n_grid_cells = 13
    n_b_boxes = 5

    # Names and colors for each class
    classes = ["yuyin", "baojing", "tft", "daozha", "xianshi", "ludeng", "jingtai", "chongdian", "jiaotong", "cheku"]
    colors = [(254.0, 0, 254), (239.88888888888889, 211.66666666666669, 127),
              (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
              (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
              (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
              (141.11111111111111, -84.66666666666664, 0), (127.0, 0, 254),
              (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
              (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
              (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254),
              (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
              (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127)]

    # Pre-computed YOLOv2 shapes of the k=5 B-Boxes
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    thresholded_predictions = []
    print('Thresholding on (Objectness score)*(Best class score) with threshold = {}'.format(score_threshold))

    # IMPORTANT: reshape to have shape = [ 13 x 13 x (5 B-Boxes) x (4 Coords + 1 Obj score + 20 Class scores ) ]
    # From now on the predictions are ORDERED and can be extracted in a simple way!
    # We have 13x13 grid cells, each cell has 5 B-Boxes, each B-Box have 25 channels with 4 coords, 1 Obj score , 20 Class scores
    # E.g. predictions[row, col, b, :4] will return the 4 coords of the "b" B-Box which is in the [row,col] grid cell
    # print(predictions.shape[3])
    predictions = np.reshape(predictions, (13, 13, 5, 15))

    # IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
    for row in range(n_grid_cells):
        for col in range(n_grid_cells):
            for b in range(n_b_boxes):

                tx, ty, tw, th, tc = predictions[row, col, b, :5]

                # IMPORTANT: (416 img size) / (13 grid cells) = 32!
                # YOLOv2 predicts parametrized coordinates that must be converted to full size
                # final_coordinates = parametrized_coordinates * 32.0 ( You can see other EQUIVALENT ways to do this...)
                center_x = (float(col) + sigmoid(tx)) * 32.0
                center_y = (float(row) + sigmoid(ty)) * 32.0

                roi_w = np.exp(tw) * anchors[2 * b + 0] * 32.0
                roi_h = np.exp(th) * anchors[2 * b + 1] * 32.0

                final_confidence = sigmoid(tc)

                # Find best class
                class_predictions = predictions[row, col, b, 5:]
                class_predictions = softmax(class_predictions)

                class_predictions = tuple(class_predictions)
                best_class = class_predictions.index(max(class_predictions))
                best_class_score = class_predictions[best_class]

                # Compute the final coordinates on both axes
                left = int(center_x - (roi_w / 2.))
                right = int(center_x + (roi_w / 2.))
                top = int(center_y - (roi_h / 2.))
                bottom = int(center_y + (roi_h / 2.))

                if ((final_confidence * best_class_score) > score_threshold):
                    thresholded_predictions.append(
                        [[left, top, right, bottom], final_confidence * best_class_score, classes[best_class]])

    # Sort the B-boxes by their final score
    thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)
    if len(thresholded_predictions) == 0 :
        return input_image


    # Non maximal suppression
    nms_predictions = non_maximal_suppression(thresholded_predictions, iou_threshold)

    # Print survived b-boxes
    for i in range(len(nms_predictions)):
        print('B-Box {} : {}'.format(i + 1, nms_predictions[i]))

    # Draw final B-Boxes and label on input image
    for i in range(len(nms_predictions)):
        color = colors[classes.index(nms_predictions[i][2])]
        best_class_name = nms_predictions[i][2]
        score = str(nms_predictions[i][1]*100)[:4]
        labels = best_class_name + " " + score +"%"
        start_x = int(nms_predictions[i][0][0]*output_width/416)
        start_y = int(nms_predictions[i][0][1]*output_height/416)
        end_x = int(nms_predictions[i][0][2]*output_width/416)
        end_y = int(nms_predictions[i][0][3]*output_height/416)

        # Put a class rectangle with B-Box coordinates and a class label on the image
        input_image = cv2.rectangle(input_image, (start_x, start_y), (end_x, end_y), color,5)
        input_image = cv2.rectangle(input_image, (start_x-3, start_y-35), (start_x+len(labels)*19, start_y), color, -1)
        cv2.putText(input_image, labels, (start_x,start_y-8),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    return input_image

def recognize(jpg_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input:0")
            print(input_x)
            #out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            #print(out_softmax)
            out_label = sess.graph.get_tensor_by_name("output:0")
            print(out_label)

            img = io.imread(jpg_path)
            img = transform.resize(img, (416, 416, 3))
            img_out = sess.run(out_label, feed_dict={input_x:np.reshape(img, [-1, 416, 416, 3])})
    return img_out

### MAIN ##############################################################################################################

def main(_):
    # Definition of the paths
    #input_img_path = './chong.jpg'
    input_img_path = '../bkrc_v2/'
    output_image_path = './output.jpg'
    model = '.model/bkrc_v2.pb'

    # Definition of the parameters
    score_threshold = 0.3
    iou_threshold = 0.3

    video_capture = cv2.VideoCapture(0)

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(model, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            input_x = sess.graph.get_tensor_by_name("input:0")
            out_label = sess.graph.get_tensor_by_name("output:0")

            while(1):
                _, frame = video_capture.read()
                img = transform.resize(frame, (416, 416, 3))
                img_out = sess.run(out_label, feed_dict={input_x: np.reshape(img, [-1, 416, 416, 3])})
                output_image = postprocessing(img_out, frame, score_threshold, iou_threshold,
                                              480, 640)
                cv2.imshow("Predictions", output_image)
                cv2.waitKey(1)


if __name__ == '__main__':
    tf.app.run(main=main)
