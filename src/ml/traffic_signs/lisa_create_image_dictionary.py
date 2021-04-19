# USAGE
# python lisa_create_images.py

# import the necessary packages
from config import lisa_config_for_create_images as config
from sklearn.model_selection import train_test_split
import os
import json

def main():
    # open the classes output file
    f = open(config.CLASSES_FILE, "w")

    print("wrote these class labels: ")
    # loop over the classes
    for (k, v) in config.CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                                    "\tname: '" + k + "'\n"
                                                      "}\n")
        f.write(item)
        print(item)
    print("into filename: ", config.CLASSES_FILE)

    # close the output classes file
    f.close()


    # initialize a data dictionary used to map each image filename
    # to all bounding boxes associated with the image, then load
    # the contents of the annotations file
    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    # loop over the individual rows, skipping the header
    # counter = 0
    for row in rows[1:]:
        # break the row into components
        row = row.split(",")[0].split(";")
        (imagePath, label, startX, startY, endX, endY, _) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        # if we are not interested in the label, ignore it
        if label not in config.CLASSES:
            continue

        # build the path to the input image, then grab any other
        # bounding boxes + labels associated with the image
        # path, labels, and bounding box lists, respectively
        p = os.path.sep.join([config.BASE_PATH, imagePath])
        b = D.get(p, [])

        # build a tuple consisting of the label and bounding box,
        # then update the list and store it in the dictionary
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b
        # counter += 1
        # if counter > 10:
        #    break

    # create an all images dictionary
    json.dump(D, open(config.ALL_IMAGES_DICTIONARY, 'w'))
    print("wrote the image dictionary as a json file: ", config.ALL_IMAGES_DICTIONARY)

    '''
    # to read, use the following
    data = json.load(open(config.ALL_IMAGES_DICTIONARY))
    print("there are ", len(data), " entries in the image dictionary")
    data_list = list(data.keys())
    for i in range(len(data_list)):
        print("image ", i, " filename: ", data_list[i])
        print("label: ", data[data_list[i]] [0][0])
        print("bounding box: ", data[data_list[i]] [0][1])

    print("first key: ", data_list[0])
    print("first label: ",data[data_list[0]] [0][0])
    print("first bounding box: ", data[data_list[0]] [0][1])
    print("second key: ", data_list[1])
    print("second label: ", data[data_list[1]] [0][0])
    print("second bounding box: ", data[data_list[1]] [0][1])
    
    # create training and testing splits dictionaries
    (trainKeys, testKeys) = train_test_split(list(D.keys()),
        test_size=config.TEST_SIZE, random_state=42)

    train_dictionary = {}
    for key_value in trainKeys:
        train_dictionary[key_value] = D[key_value]
    json.dump(train_dictionary, open(config.TRAIN_IMAGES_DICTIONARY, 'w'))

    test_dictionary = {}
    for key_value in testKeys:
        test_dictionary[key_value] = D[key_value]
    json.dump(test_dictionary, open(config.TEST_IMAGES_DICTIONARY, 'w'))

    
    # to read, use the following
    training_data = json.load(open(config.TRAIN_IMAGES_DICTIONARY))
    print("there are ", len(training_data), " entries in the image dictionary")
    training_data_list = list(training_data.keys())
    for i in range(len(training_data_list)):
        print("training image ", i, " filename: ", training_data_list[i])
        print("label: ", training_data[training_data_list[i]] [0][0])
        print("bounding box: ", training_data[training_data_list[i]] [0][1])

    print("first training key: ", training_data_list[0])
    print("first training label: ",training_data[training_data_list[0]] [0][0])
    print("first training bounding box: ", training_data[training_data_list[0]] [0][1])
    print("second training key: ", training_data_list[1])
    print("second training label: ", training_data[training_data_list[1]] [0][0])
    print("second training bounding box: ", training_data[training_data_list[1]] [0][1])
    


    '''
# check to see if the main thread should be started
if __name__ == "__main__":
    # tf.app.run()
    main()
