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
        self.classifier = None
        self.encoder = None

    # returns all embeddings
    def obtainImageEmbedding(self, face):
        face = (face / 255.).astype(np.float32)
        embedding = self.nn_model.predict(np.expand_dims(face, axis=0))[0]
    
        return embedding

    # used to load the svm classifier
    def loadSVMClassifier(self):
        # load known embeddings
        print("Loading svm classifier...\nLoading known embeddings.")
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

        self.classifier = svc
        self.encoder = encoder

    # used to load the knn classifier
    def loadKNNClassifier(self):
        # load known embeddings
        print("Loading knn classifier...\nLoading known embeddings.")
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

        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        knn.fit(X_train, Y_train)

        accuracy = accuracy_score(Y_test, knn.predict(X_test))
        print("Accuracy = {}".format(accuracy))

        self.classifier = knn
        self.encoder = encoder

    # classify one embedding
    def classify(self, emb):
        if self.classifier is None:
            print("No classifier is loaded!")
            return 

        prediction = self.classifier.predict([emb])
        identity = self.encoder.inverse_transform(prediction)[0]

        #TODO - check for "unknown" person
        # very big distance or something?

        return identity



# used to detect the faces (and their landmarks) from a photo and align them
# using dlib and OpenFace lib
def detectAndAlignFaces(image, align, testing=False):
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

# returns the identities of the recognized faces and their bounding boxes
def recognizeFacesInImage(img, fr, align):
    print("Detecting and aligning face(s).")
    faces, boxes = detectAndAlignFaces(img, align, testing=False)
    if faces is None:
        print("No face detected in the picture.")
        return [], []

    # obtain embedding
    identities = []
    print("Obtaining embedding(s) and classify the face(s).")
    for face in faces:
        emb = fr.obtainImageEmbedding(face)
        identities.append(fr.classify(emb))

    if len(identities) != len(faces) or len(identities) == 0 or len(faces) == 0:
        print("No faces were recognized in this frame.")
        return [], []

    return identities, boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-embeddings", type=str, help="prints the embeddings of the face(s) in the picture")
    parser.add_argument("--generate-embeddings", help="generates embeddings for the images folder")
    parser.add_argument("--align-faces", type=str, help="used to detect and align faces of an image")

    parser.add_argument("--classify-knn", type=str, help="used to classify a face using knn")
    parser.add_argument("--classify-svm", type=str, help="used to classify a face using svm")

    parser.add_argument("--webcam", action="store_true", help="recognize faces from the webcam(if the webcam is available)")

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
        fr.loadKNNClassifier()
        image = cv2.imread(args.classify_knn)
        identities, boxes = recognizeFacesInImage(image, fr, align)

        # draw boxes over the recognized faces and label them
        for indx, label in enumerate(identities):
            left, top, right, bottom = boxes[indx].left(), boxes[indx].top(), boxes[indx].right(), boxes[indx].bottom()
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # show the output image
        cv2.imshow("Faces", image)
        cv2.waitKey(0)

    elif args.classify_svm:
        fr.loadSVMClassifier()
        image = cv2.imread(args.classify_svm)
        identities, boxes = recognizeFacesInImage(image, fr, align)

        # draw boxes over the recognized faces and label them
        for indx, label in enumerate(identities):
            left, top, right, bottom = boxes[indx].left(), boxes[indx].top(), boxes[indx].right(), boxes[indx].bottom()
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # show the output images
        cv2.imshow("Faces", image)
        cv2.waitKey(0)

    elif args.webcam:
        # video capture code from https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
        video_capture = cv2.VideoCapture(0)

        if video_capture is None or not video_capture.isOpened():
            print("Error opening webcam.")
            return

        fr.loadSVMClassifier()

        frame_no = 1
        while True:
            if frame_no != 4:
                frame_no += 1
                continue
            else:
                frame_no = 1

            # Grab a single frame of video and resize
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # process frame
            identities, boxes = recognizeFacesInImage(small_frame, fr, align)

            # when lens are == 0 skip the classification
            if len(identities) > 0 or len(boxes) > 0:
                # draw boxes over the recognized faces and label them
                for indx, label in enumerate(identities):
                    left, top, right, bottom = boxes[indx].left() * 4, boxes[indx].top() * 4, boxes[indx].right() * 4, boxes[indx].bottom() * 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()