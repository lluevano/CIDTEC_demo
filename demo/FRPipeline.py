import cv2
import numpy as np
import mxnet as mx
import insightface
import pickle
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances

class FRP:  # Face Recognition Pipeline
    # define constants for face detection algorithms
    DETECT_HAAR = 0
    DETECT_MOBILE_MNET = 1

    # define constants for alignment
    ALIGN_INSIGHTFACE = 0
    ALIGNMENT_RESOLUTION = 112

    # define constants for face recognition algorithms
    RECOGNITION_SHUFFLEFACENET = 0

    def __init__(self, use_gpu, detection_algorithm, alignment_algorithm, recognition_algorithm, recognition_batch_size=4, db_filename=None):
        # PREPARE USE CPU OR GPU
        assert isinstance(use_gpu, bool)
        self.use_gpu = -1 if not use_gpu else 0
        self.biometric_templates = None
        self.labels = None

        # LOAD DETECTION MODEL
        self.active_detection_model = detection_algorithm  # algorithm number
        self.detection_model = self.prepare_detection(self.active_detection_model)  # model object

        # SET ALIGNNMENT METHOD
        self.active_alignment_method = alignment_algorithm

        # LOAD RECOGNITION MODEL
        self.active_recognition_model = recognition_algorithm
        self.recognition_model = self.prepare_recognition(self.active_recognition_model, recognition_batch_size, db_filename)

        self.identification_threshold = 0

    # INITIALIZE MODELS
    def prepare_detection(self, active_detection_model):
        if active_detection_model == self.DETECT_HAAR:
            model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        elif active_detection_model == self.DETECT_MOBILE_MNET:
            model = insightface.model_zoo.get_model('retinaface_mnet025_v2')
            model.prepare(ctx_id=self.use_gpu, nms=0.4)
        return model

    def prepare_recognition(self, active_recognition_model, recognition_batch_size=1, db_filename=None):
        # load database
        if db_filename:
            with open(db_filename, 'rb') as db_f_handler:
                db_pickle = pickle.load(db_f_handler)
                self.biometric_templates = db_pickle['biometric_templates']
                self.labels = db_pickle['labels']

        if active_recognition_model == self.RECOGNITION_SHUFFLEFACENET:
            sym, arg_params, aux_params = mx.model.load_checkpoint('./models/shufflev2-1.5-arcface-retina/model', 42)
            model = mx.mod.Module(symbol=sym, context=[mx.cpu() if self.use_gpu == -1 else mx.gpu(0)], label_names=None)
            model.bind(data_shapes=[('data', (recognition_batch_size, 3, 112, 112))])
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
    def alignCropFace(self, face_image, landmarks):
        if self.active_alignment_method == self.ALIGN_INSIGHTFACE:
            aligned_face = insightface.utils.face_align.norm_crop(face_image, landmarks, image_size=FRP.ALIGNMENT_RESOLUTION)
        else:
            raise "Selected alignment method not implemented"
        return aligned_face

    def batch_extract_norm_embeds(self, aligned_faces, format="RGB", scale_faces=True):
        # receives aligned face and returns similarity index against database
        if self.active_recognition_model == self.RECOGNITION_SHUFFLEFACENET:
            from MxNetEmbedExtractor import MxNetEmbedExtractor

            aligned_faces = ((aligned_faces - 127.5) * 0.0078125) if scale_faces else aligned_faces

            extractor = MxNetEmbedExtractor(self.recognition_model)
            embeddings = extractor.extract_batch_embedding(aligned_faces, color_format=format, batch_size=8)

            #L2 NORM
            embeddings = normalize(embeddings)
            #embeddings = embeddings / np.reshape(np.linalg.norm(embeddings, axis=1, ord=2), (embeddings.shape[0], 1))

        else:
            raise f"{self.active_recognition_model} Not yet implemented"

        return embeddings

    def match_scores(self, embeddings):
        #scores = np.dot(embeddings, self.biometric_templates.T)/(np.linalg.norm(embeddings)*np.linalg.norm(self.biometric_templates))
        scores = 1 - pairwise_distances(embeddings,self.biometric_templates, metric='cosine')
        matches = np.argmax(scores, axis=1)

        # TODO: Time-profile best choice
        final_scores = np.fromiter((row[index] for row, index in
                                    zip(scores, matches)), dtype=float)

        final_ids = np.fromiter(
            (index if row[index] >= self.identification_threshold else -1 for row, index in
             zip(scores, matches)), dtype=int)

        return final_scores, final_ids
    def executeFullPipeline(self, img, format="RGB"):
        if format=="BGR":
            img_rgb = np.ascontiguousarray(img[..., ::-1])
        else:
            assert format=="RGB"
            img_rgb = img

        #DETECTION
        bbox, lmk = self.detectFaces(img_rgb)

        #ALIGNMENT
        n_faces = np.shape(bbox)[0]

        if not n_faces:
            return bbox, None, None, None, None

        aligned_faces=np.zeros((n_faces, FRP.ALIGNMENT_RESOLUTION, FRP.ALIGNMENT_RESOLUTION, 3),dtype=np.float32)
        for i in range(n_faces):
            #corner1 = (bbox[i][0], bbox[i][1])  # (x,y)
            #corner2 = (bbox[i][2], bbox[i][3])
            #face_img = img_rgb[corner1[1]:corner2[1], corner1[0]:corner2[0], :]
            aligned_faces[i, ...] = cv2.cvtColor(self.alignCropFace(img_rgb, lmk[i]),cv2.COLOR_RGB2BGR)

        #EMBED EXTRACTION
        embeddings = self.batch_extract_norm_embeds(aligned_faces, scale_faces=False)

        #MATCHING
        scores, ids = self.match_scores(embeddings)

        return bbox, lmk, ids, scores, embeddings
