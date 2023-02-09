import cv2
import numpy as np
import mxnet as mx
import insightface
import pickle

class FRP:  # Face Recognition Pipeline
    # define constants for face detection algorithms
    DETECT_HAAR = 0
    DETECT_MOBILE_MNET = 1

    # define constants for alignment
    ALIGN_INSIGHTFACE = 0
    ALIGNMENT_RESOLUTION = 112

    # define constants for face recognition algorithms
    RECOGNITION_SHUFFLEFACENET = 0

    def __init__(self, use_gpu, detection_algorithm, alignment_algorithm, recognition_algorithm):
        # PREPARE USE CPU OR GPU
        assert isinstance(use_gpu, bool)
        self.use_gpu = -1 if not use_gpu else 0
        self.identity_vectors = None
        self.labels = None

        # LOAD DETECTION MODEL
        self.active_detection_model = detection_algorithm  # algorithm number
        self.detection_model = self.prepare_detection(self.active_detection_model)  # model object

        # SET ALIGNNMENT METHOD
        self.active_alignment_method = alignment_algorithm

        # LOAD RECOGNITION MODEL
        self.active_recognition_model = recognition_algorithm
        self.recognition_model = self.prepare_recognition(self.active_recognition_model)

        self.identification_threshold = 0.75

    # INITIALIZE MODELS
    def prepare_detection(self, active_detection_model):
        if active_detection_model == self.DETECT_HAAR:
            model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        elif active_detection_model == self.DETECT_MOBILE_MNET:
            model = insightface.model_zoo.get_model('retinaface_mnet025_v2')
            model.prepare(ctx_id=self.use_gpu, nms=0.4)
        return model

    def prepare_recognition(self, active_recognition_model):
        # load database
        with open('test_database.pickle', 'rb') as test_emb_file:
            self.identity_vectors = pickle.load(test_emb_file)

        with open('labels.pickle', 'rb') as label_file:
            self.labels = pickle.load(label_file)

        if active_recognition_model == self.RECOGNITION_SHUFFLEFACENET:
            sym, arg_params, aux_params = mx.model.load_checkpoint('./models/shufflev2-1.5-arcface-retina/model', 42)
            model = mx.mod.Module(symbol=sym, context=[mx.cpu() if self.use_gpu == -1 else mx.gpu(0)], label_names=None)
            model.bind(data_shapes=[('data', (1, 3, 112, 112))])
            model.set_params(arg_params, aux_params)
        else:
            raise f"Model {active_recognition_model} not implemented."
        return model

    # PERFORM DETECTION
    def detectFaces(self, image):
        if self.active_detection_model == FRP.DETECT_HAAR:
            faces = self.detection_model.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.1, 4)
            bbox = [(x, y, x + w, y + h) for x, y, w, h in faces]
            # TODO: get 5 landmarks for haarcascades missing

        elif self.active_detection_model == FRP.DETECT_MOBILE_MNET:
            # bbox: tuple with 2 opposite rectangle corners where the face is located
            # landmark: five landmarks of the detected face: 2 eyes, tip of the nose, and 2 lip opposite edges
            # insightface format
            bbox, landmarks = self.detection_model.detect(image, threshold=0.5, scale=1.0)

        bbox_int = bbox.astype(np.int)
        landmark_int = landmarks.astype(np.int)

        return bbox_int, landmark_int

    # PERFORM SINGLE FACE ALIGNMENT
    def alignFace(self, face_image, landmarks):
        if self.active_alignment_method == self.ALIGN_INSIGHTFACE:
            aligned_face = insightface.utils.face_align.norm_crop(face_image, landmarks, image_size=FRP.ALIGNMENT_RESOLUTION)
        else:
            return None
        return aligned_face

    # PERFORM SINGLE FACE RECOGNITION
    def recognizeSingleFace(self, aligned_face, format="RGB"):
        # receives aligned face and returns similarity index against database
        if self.active_recognition_model == self.RECOGNITION_SHUFFLEFACENET:

            im_tensor = np.zeros((1, 3, aligned_face.shape[0], aligned_face.shape[1]))
            for i in range(3):
                if format=="RGB":
                    offset = i
                elif format=="BGR":
                    offset = 2 - i
                else:
                    raise "Unknown image format"
                im_tensor[0, i, :, :] = aligned_face[:, :, offset]

            im_tensor = (im_tensor - 127.5) * 0.0078125
            data = mx.ndarray.array(im_tensor)
            db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
            self.recognition_model.forward(db, is_train=False)

            # Normalize embedding obtained from forward pass to unit vector
            embedding = self.recognition_model.get_outputs()[0].squeeze()
            embedding /= embedding.norm()
            # sim = np.dot(embedding, test_embedding.asnumpy().T)
            sim_vector = np.dot(embedding.asnumpy(), self.identity_vectors.T)

            best_match = np.argmax(sim_vector)
            # print("BEST MATCH FOUND AT")
            # print(sim_vector)
            # print(best_match)
            sim = sim_vector[best_match]

        return sim, best_match

    def recognizeBatchFace(self, aligned_faces, format="RGB", batch_size=8):
        # receives aligned face and returns similarity index against database
        if self.active_recognition_model == self.RECOGNITION_SHUFFLEFACENET:
            from MxNetEmbedExtractor import MxNetEmbedExtractor

            aligned_faces = (aligned_faces - 127.5) * 0.0078125

            extractor = MxNetEmbedExtractor(self.recognition_model)
            embeddings = extractor.extract_batch_embedding(aligned_faces, format, batch_size=8)

            # sim = np.dot(embedding, test_embedding.asnumpy().T)
            sim_vector = np.dot(embeddings, self.identity_vectors.T)

            best_match = np.argmax(sim_vector)
            # print("BEST MATCH FOUND AT")
            # print(sim_vector)
            # print(best_match)
            sim = sim_vector[best_match]

        return sim, best_match

    def executeFullPipeline(self, img):
        #Assumes RGB format
        bbox, lmk = self.detectFaces(img)
        n_faces = np.shape(bbox)[0]

        aligned_faces=np.zeros((n_faces, FRP.ALIGNMENT_RESOLUTION, FRP.ALIGNMENT_RESOLUTION, 3))
        for i in range(n_faces):
            corner1 = (bbox[i][0], bbox[i][1])  # (x,y)
            corner2 = (bbox[i][2], bbox[i][3])
            face_img = img[corner1[1]:corner2[1], corner1[0]:corner2[0], :]
            aligned_faces[i, ...] = self.alignFace(face_img, lmk[i])

        embeddings = self.recognizeBatchFace(aligned_faces)
        scores = np.dot(embeddings, self.identity_vectors.T)
        matches = np.argmax(scores, axis=1)
        np.fromiter((row[index] if row[index] >= self.identification_threshold else -1 for row, index in zip(scores, matches)), dtype=float)

        return embeddings, bbox, lmk