import sys, dlib, cv2, os, pprint, openface, json, argparse, random
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
MODEL_PATH = "./data/nn4.small2.v1.h5"
IMAGES_PATH = "./images/"

class FaceRecognizer():
    def __init__(self):
        with CustomObjectScope({'tf': tf}):
            nn_model = load_model(MODEL_PATH)
        self.nn_model = nn_model

    # returns all embeddings
    def obtainImageEmbedding(self, face):
        face = (face / 255.).astype(np.float32)
        embedding = self.nn_model.predict(np.expand_dims(face, axis=0))[0]
    
        return embedding


# used to detect the faces (and their landmarks) from a photo and align them
# using dlib and OpenFace lib
def detectAndAlignFaces(imgPath, align, testing=False):
    image = cv2.imread(imgPath)
    alignedFaces = []

    # obtain faces bounding boxes
    boundingBoxes = align.getAllFaceBoundingBoxes(image) 

    if boundingBoxes is None:
        print("No faces were detected.")
        return

    # for each bounding box extract and align the face in it
    for i, box in enumerate(boundingBoxes):
        alignedFace = align.align(96, image, box, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        alignedFaces.append(alignedFace)

        if testing:
            cv2.imshow("Aligned face {}".format(i), alignedFace)
            cv2.waitKey(0)

    return alignedFaces, boundingBoxes


# go through the training folder and compute the embeddings and save as json
def saveTrainingImagesEmbeddings(fr, align):
    for d in (x for x in os.listdir(IMAGES_PATH) if not x.endswith(".json")):
        fname = "{}.json".format(os.path.join(IMAGES_PATH, d))

        # if there are no embeddings for the current person obtain them
        if not os.path.exists(fname):
            print("Generating embeddings for {}".format(d))
            with open(fname, "w") as f:
                embeddings_list = []

                for img in os.listdir(os.path.join(IMAGES_PATH, d)):
                    imgPath = os.path.join(IMAGES_PATH, d, img)
                    faces, _ = detectAndAlignFaces(imgPath, align)

                    if faces is None or len(faces) == 0:
                        print ("No face detected in {}.".format(imgPath))
                        print ("No embedding generated for {}.".format(imgPath))
                        continue

                    # there's only one face anyway
                    e = fr.obtainImageEmbedding(faces[0])

                    if e is None:
                        print ("Face detected but no embedding generated for {}.".format(imgPath))
                    else:
                        # append as list not numpy (bc it's not json serializable)
                        embeddings_list.append(e.tolist())

                json.dump(embeddings_list, f)

# load training folder pictures embeddings
def loadTrainingImagesEmbeddings():
    data = {}

    for json_file in (x for x in os.listdir(IMAGES_PATH) if x.endswith(".json")):
        label = json_file[:-5]
        print("Loading embeddings for {}.".format(label))

        with open(os.path.join(IMAGES_PATH, json_file), "r") as f:
            data[label] = np.array(json.loads(f.read()))

    return data

# classify using KNN
def classifyKNN(fr, align, imgPath):
    print("Detecting and aligning face.")
    faces, _ = detectAndAlignFaces(imgPath, align, testing=False)
    if faces is None or len(faces) != 1:
        print("No face (or more than oen face) detected in the picture.")
        return

    # obtain embedding
    print("Obtaining embedding.")
    emb = fr.obtainImageEmbedding(faces[0])

    # load known embeddings
    print("Loading known embeddings.")
    data = loadTrainingImagesEmbeddings()

    # have fun
    targets = ([k for k in data])

    print(targets)

    encoder = LabelEncoder()
    encoder.fit(targets)
    y = encoder.transform(targets)

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    # split the data 70% training 30% testing
    for label in data:

        # randomly pick 30%
        no = int(0.3 * len(data[label]))
        indxs = list(range(0,len(data[label])))

        # obtain random indexes
        rand_indexes = random.sample(indxs, no)

        for i, embd in enumerate(data[label]):
            if i in rand_indexes:
                X_test.append(embd)
                Y_test.extend(encoder.transform([label]))
            else:
                X_train.append(embd)
                Y_train.extend(encoder.transform([label]))

    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X_train, Y_train)

    accuracy = accuracy_score(Y_test, knn.predict(X_test))

    print("Accuracy = {}".format(accuracy))

    prediction = knn.predict([emb])
    identity = encoder.inverse_transform(prediction)[0]

    image = cv2.imread(imgPath)
    cv2.imshow("{}".format(identity), image)
    cv2.waitKey(0)

# classify using SVM
def classifySVM(fr, align, imgPath):
    print("Detecting and aligning face.")
    faces, _ = detectAndAlignFaces(imgPath, align, testing=False)
    if faces is None or len(faces) != 1:
        print("No face (or more than oen face) detected in the picture.")
        return

    # obtain embedding
    print("Obtaining embedding.")
    emb = fr.obtainImageEmbedding(faces[0])

    # load known embeddings
    print("Loading known embeddings.")
    data = loadTrainingImagesEmbeddings()

    # have fun
    targets = ([k for k in data])

    print(targets)

    encoder = LabelEncoder()
    encoder.fit(targets)
    y = encoder.transform(targets)

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    # split the data 70% training 30% testing
    for label in data:

        # randomly pick 30%
        no = int(0.3 * len(data[label]))
        indxs = list(range(0,len(data[label])))

        # obtain random indexes
        rand_indexes = random.sample(indxs, no)

        for i, embd in enumerate(data[label]):
            if i in rand_indexes:
                X_test.append(embd)
                Y_test.extend(encoder.transform([label]))
            else:
                X_train.append(embd)
                Y_train.extend(encoder.transform([label]))

    svc = SVC() # default kernel is rbf 
    svc.fit(X_train, Y_train)

    accuracy = accuracy_score(Y_test, svc.predict(X_test))

    print("Accuracy = {}".format(accuracy))

    prediction = svc.predict([emb])
    identity = encoder.inverse_transform(prediction)[0]

    image = cv2.imread(imgPath)
    cv2.imshow("{}".format(identity), image)
    cv2.waitKey(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-embeddings", type=str, help="prints the embeddings of the face(s) in the picture")
    parser.add_argument("--generate-embeddings", help="generates embeddings for the images folder")
    parser.add_argument("--align-faces", type=str, help="used to detect and align faces of an image")

    parser.add_argument("--classify-knn", type=str, help="used to classify a face using knn")
    parser.add_argument("--classify-svm", type=str, help="used to classify a face using svm")

    args = parser.parse_args()
    
    if len(sys.argv) < 2:
        parser.print_help()
        return

    # init recognizer and face aligne
    fr = FaceRecognizer()
    align = openface.AlignDlib(PREDICTOR_PATH)

    if args.print_embeddings:
        faces, _ = detectAndAlignFaces(args.print_embeddings, align, testing=False)
        if faces is None or len(faces) == 0:
            print("No faces detected in the picture.")
            return

        for i, face in enumerate(faces):
            print("Embeddings for face {}:".format(i))
            embedding = fr.obtainImageEmbedding(face)
            pprint.pprint(embedding)

    elif args.generate_embeddings:
        print("Generating embeddings:")
        saveTrainingImagesEmbeddings(fr, align)
    
    elif args.align_faces:
        faces, _ = detectAndAlignFaces(args.align_faces, align, testing=True)
        if faces is None or len(faces) == 0:
            print("No faces detected in the picture.")
            return

    elif args.classify_knn:
        classifyKNN(fr, align, args.classify_knn)

    elif args.classify_svm:
        classifySVM(fr, align, args.classify_svm)

if __name__ == "__main__":
    main()