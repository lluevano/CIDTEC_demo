import mxnet as mx
import numpy as np

class MxNetEmbedExtractor:
    def __init__(self, model):
        self.model = model
        self.feat_size = self.model.get_outputs()[0].shape[-1]

    def extract_batch_embedding(self, img_batch: np.array, color_format: str, batch_size=2):
        #  Receives a NxHxWxC numpy array
        #  Returns a NxEMB_SIZE list
        if color_format == "BGR":
            img_batch = np.ascontiguousarray(img_batch[...,::-1]) #convert to RGB
        else:
            assert color_format == "RGB", "Only RGB and BGR formats are supported"

        img_batch = np.transpose(img_batch, (0, 3, 1, 2)).astype(np.float32) # to NxCxHxW

        feat_size = self.feat_size

        data_iter = mx.io.NDArrayIter(img_batch, batch_size=batch_size)
        final_embeds = np.zeros((img_batch.shape[0], feat_size))
        remainder = np.shape(img_batch)[0] % batch_size
        total_batches = np.ceil(np.shape(img_batch)[0] / batch_size)
        tail = None
        for i, batch in enumerate(data_iter):
            self.model.forward(batch, is_train=False)
            embeddings = self.model.get_outputs()[0]
            if (i+1) == total_batches:
                final_tail = (i*batch_size)+remainder if remainder else None
                tail = remainder if remainder else None
            else:
                final_tail = (i*batch_size)+batch_size
            final_embeds[i * batch_size:final_tail, :] = embeddings.asnumpy()[:tail, :]
        #final_embeds = final_embeds[:-remainder if remainder else None]
        return final_embeds
