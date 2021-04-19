

import json
import cv2
from config import lisa_config_for_create_images as config

def main():
    images = get_images(config.ALL_IMAGES_DICTIONARY)
    counter = 0
    for image in images:
        #cv2.imshow("image", image[2])
        #cv2.waitKey(0)
        #filename = image[0]
        label = image[1]
        cropped_image = image[2]
        # image_filename = config.CROPPED_IMAGES + "/" + label + "/" + filename
        image_filename = config.CROPPED_IMAGES + "/" + label + "/" + label + "_" + str(counter) + ".png"
        counter += 1
        cv2.imwrite(image_filename, cropped_image)
        print("cropped image written to: ", image_filename)

    '''
    training_images = get_images(config.TRAIN_IMAGES_DICTIONARY)
    counter = 0
    for image in training_images:
        #cv2.imshow("image", image[2])
        #cv2.waitKey(0)
        filename = image[0]
        label = image[1]
        cropped_image = image[2]
        # image_filename = config.CROPPED_IMAGES + "/" + label + "/" + filename
        image_filename = config.CROPPED_IMAGES + "/training/" + label + "/" + label + "_" + str(counter) + ".png"
        counter += 1
        cv2.imwrite(image_filename, cropped_image)
        print("cropped image written to: ", image_filename)

    testing_images = get_images(config.TEST_IMAGES_DICTIONARY)
    counter = 0
    for image in testing_images:
        #cv2.imshow("image", image[2])
        #cv2.waitKey(0)
        filename = image[0]
        label = image[1]
        cropped_image = image[2]
        # image_filename = config.CROPPED_IMAGES + "/" + label + "/" + filename
        image_filename = config.CROPPED_IMAGES + "/testing/" + label + "/" + label + "_" + str(counter) + ".png"
        counter += 1
        cv2.imwrite(image_filename, cropped_image)
        print("cropped image written to: ", image_filename)
    '''

def get_images(filename):
    image_list = []
    data = json.load(open(filename))
    print("there are ", len(data), " entries in the ", filename, " image dictionary")
    data_list = list(data.keys())
    for i in range(len(data_list)):
        print("image ", i, " filename: ", data_list[i])
        print("label: ", data[data_list[i]][0][0])
        print("bounding box: ", data[data_list[i]][0][1])
        img = cv2.imread(data_list[i])
        image_list.append([data_list[i], data[data_list[i]][0][0], crop_image(img, data[data_list[i]][0][1])])
    return image_list

def crop_image(img, bounding_box):
    startX, startY, endX, endY = bounding_box
    return img[int(startY):int(endY), int(startX):int(endX)]

# check to see if the main thread should be started
if __name__ == "__main__":
    # tf.app.run()
    main()