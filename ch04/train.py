import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
### use GPU
# config.GPU = True
import pickle
from cbow import CBOW
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import create_contexts_target, to_cpu
from dataset import ptb


# high parameter
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# load data
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
### use GPU
# if config.GPU:
#     contexts, target = to_gpu(contexts), to_gpu(target)

# create model
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# start train
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 後から利用できる様に、必要なデータを保存
word_vecs = model.word_vecs
### use GPU
# if config.GPU:
#   word_vecs = to_cpu(word_vecs)
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)


