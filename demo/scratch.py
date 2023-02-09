from MxNetEmbedExtractor import MxNetEmbedExtractor
import mxnet as mx
import numpy as np


sym, arg_params, aux_params = mx.model.load_checkpoint('./models/shufflev2-1.5-arcface-retina/model', 42)

imgs = np.ones((5,112,112,3))
for i in range(5):
    imgs[i] = imgs[i]*(i+1)

model = mx.mod.Module(symbol=sym, context=[mx.cpu()], label_names=None)
model.bind(data_shapes=[('data', (imgs.shape[0], 3, 112, 112))])
model.set_params(arg_params, aux_params)

extractor = MxNetEmbedExtractor(model,mx.cpu())
embeddings = extractor.extract_batch_embedding(imgs,"RGB",batch_size=6)

print("done")
