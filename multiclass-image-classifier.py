import numpy as np  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt  
import math
import h5py as h5py

# not strictly necessary - used to display the result of a prediction
import cv2  

# image dimensions:
img_width, img_height = 224, 224  

# file paths:
top_model_weights_path = 'bottleneck_fc_model.h5'  
train_data_dir = 'data/train'  
validation_data_dir = 'data/validation'  

# number of epochs to train the top model:
epochs = 50  
# batch size used by flow_from_directory and predict_generator:  
batch_size = 16 

def save_bottlebeck_features():
    # build the VGG16 network, pre-trained on imagenet, but without the
    # final fully-connected layers (specified by include_top = False)
    model = applications.VGG16(include_top = False,
                               weights = 'imagenet')

    datagen = ImageDataGenerator(rescale = 1./255)

    # takes a path to a directory and generates batches of data
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)
    
    print(nb_train_samples)
    print(generator.class_indices)
    print(num_classes)

    # number of batches to train on
    # necessary because of a bug in predict_generator where it can't 
    # determine the correct number of iterations when the number of 
    # training samples isn't divisible by the batch size
    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    # generate predictions by running generator on VGG16 model to get
    # bottleneck features for training
    print("Running generator on VGG16 model for training...")
    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)
    
    print("Saving training bottleneck features...")
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    # do the same as above, but for the validation samples
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    print("Running generator on VGG16 model for validation...")
    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    print("Saving validation bottleneck features...")
    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)

def train_top_model():
    # create a generator for fetching the class labels for each of the
    # training/validation samples (class_mode = 'categorical')
    datagen_top = ImageDataGenerator(rescale = 1./255)
    
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')

    # get the class labels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes = num_classes)

    # do the same as above for the validation features
    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation.npy')

    # convert the validation labels into a binary class matrix
    # (necessary for use with categorical_crossentropy loss function)
    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes = num_classes)

    ### THE NETWORK ###
    # create and train a small fully-connected network (the top model)
    # using the bottleneck features as input, outputs the classifier
    model = Sequential()
    model.add(Flatten(input_shape = train_data.shape[1:]))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = 'sigmoid'))

    model.compile(optimizer = 'rmsprop',
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])

    # fit the model to the training samples, evaluating the loss and metrics
    # on the validation data
    history = model.fit(train_data, train_labels,
                        epochs = epochs,
                        batch_size = batch_size,
                        validation_data = (validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels,
        batch_size = batch_size, verbose = 1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    plt.figure(1)

    # graph the training history for clarity:
    
    # summarise history for accuracy
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')

    # summarise history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()

def predict(image_path):
    # load the class_indices saved during training
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    orig = cv2.imread(image_path)

    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size = (224, 224))
    image = img_to_array(image)

    # rescale the image in the same way as the training data
    image = image / 255

    image = np.expand_dims(image, axis = 0)

    # build the VGG16 network
    model = applications.VGG16(include_top = False,
                               weights = 'imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = 'sigmoid'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    # it's easier to work with a dictionary with index as key and label
    # as value, so invert the labels map:
    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))

    # display the predictions with the image
    cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()