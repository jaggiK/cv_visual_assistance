# import the necessary packages
import os

# initialize the base path for the LISA dataset
BASE_PATH = "lisa_dataset/signDatabasePublicFramesOnly"

# build the path to the annotations file
ANNOT_PATH = os.path.sep.join([BASE_PATH, "allAnnotations.csv"])

CROPPED_IMAGES = os.path.sep.join([BASE_PATH,
	"cropped_images"])
# build the path to the output training and testing record files,
# along with the class labels file
ALL_IMAGES_DICTIONARY = os.path.sep.join([BASE_PATH,
	"records/all_images_dictionary.json"])
TRAIN_IMAGES_DICTIONARY = os.path.sep.join([BASE_PATH,
	"records/train_images_dictionary.json"])
TEST_IMAGES_DICTIONARY = os.path.sep.join([BASE_PATH,
	"records/test_images_dictionary.json"])

CLASSES_FILE = os.path.sep.join([BASE_PATH,
	"records/classes.txt"])

# initialize the test split size
TEST_SIZE = 0.25

# initialize the class labels dictionary
CLASSES = {"pedestrianCrossing": 1, "signalAhead": 2, "stop": 3}