from imutils import paths
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os



def getdataset(imagepath, setname):
    data = []
    labels = []
    print(f"Loading {setname} images...")
    ImagePaths = list(paths.list_images(basePath=imagepath))
    for imgpath in ImagePaths:
        label = imgpath.split(os.path.sep)[-2]
        img = load_img(imgpath, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)
        data.append(img)
        labels.append(label)
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    print(lb.classes_)
    print("Loadind Complete!")
    return data, labels


def createandfitmodel():

    INIT_LR = 1e-4
    EPOCHS = 20
    BS = 32

    testX, testY = getdataset("New Masks Dataset/Test", "Test")
    trainX, trainY = getdataset("New Masks Dataset/Test", "Train")
    validX, validY = getdataset("New Masks Dataset/Test", "Validation")

    basemodel = MobileNetV2(weights="imagenet", input_tensor=Input(shape=(224, 224, 3)), include_top=False)

    header = basemodel.output
    header = AveragePooling2D(pool_size=(7, 7))(header)
    header = Flatten(name="flatten")(header)
    header = Dense(128, activation="relu")(header)
    header = Dropout(0.5)(header)
    header = Dense(2, activation="softmax")(header)

    model = Model(inputs=basemodel.input, outputs=header)

    for layer in basemodel.layers:
        layer.trainable = False

    print("Compiling the model....")
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS),
                  metrics=["accuracy"])

    print("Fitting the model.....")
    Final = model.fit(trainX, trainY,
                      batch_size=BS,
                      validation_data=(validX, validY),
                      validation_steps=len(validX) // BS,
                      steps_per_epoch=len(trainY) // BS,
                      epochs=EPOCHS)

    print("Evaluating the model..")
    predY = model.predict(testX, batch_size=BS)
    predY = np.argmax(predY, axis=1)
    print(classification_report(testY.argmax(axis=1), predY, target_names=["Mask", "Non Mask"]))
    print("Saving the model...")
    model.save("model", save_format="h5")

    print("Creating and saving plots...")
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), Final.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), Final.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), Final.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), Final.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot")

createandfitmodel()