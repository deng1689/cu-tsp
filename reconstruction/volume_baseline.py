import numpy as np
import pickle
import tensorflow as tf
import keras
import keras.backend as K

from pdb import set_trace
from sklearn import preprocessing
from tqdm import tqdm
from scipy.misc import imresize
from keras.activations import relu
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, LSTM, Reshape

# CONSTANTS

AA_CORPUS = 'ACDEFGHIKLMNPQRSTVWXY'
Q8_CORPUS = 'GHITEBS-'

N_FOLDS = 1
DATA_PATH = './data/train_fold_{}.pkl'

# Load data
def load_data(path):
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data

train_pdbs = []
train_pdb_aas = []
train_q8s = []
train_dcalphas = []
train_msas = []

for i in range(1, N_FOLDS+1):
  fold_path = DATA_PATH.format(i)
  fold_data = load_data(fold_path)
  indices, pdbs, length_aas, pdb_aas, q8s, dcalphas, psis, phis, msas = fold_data
  train_pdbs += pdbs
  train_pdb_aas += pdb_aas
  train_q8s += q8s
  train_dcalphas += dcalphas
  train_msas += msas
  print('Loaded training fold {}: {} proteins'.format(i, len(pdbs)))

# Generates duplicate matrices from a column vector
def col_to_mat(x): 
  # Input: 1-by-n vector
  # Output: two n-by-n duplicate matrices

  n = len(x)
  x_orig = x
  
  for i in range(n-1): 
    x = np.hstack((x, x_orig))
  
  x_T = np.transpose(x)
  
  return x, x_T

# Converts a sequence(str) into one-hot encoding
def one_hot(seq, corpus):
    # Input: sequence(str), corpus
    # Output: n-by-L matrix (L = len(corpus))
  
    encoding = np.zeros((len(seq),len(corpus)), dtype=np.int64)
    for idx, seq_chr in enumerate(seq):
        tmp = np.zeros(len(corpus), dtype=np.int64)
        tmp[corpus.find(seq_chr)] = 1
        encoding[idx] = tmp
    return encoding

def initialize_one_hot_encoder(corpus):
    """
    corpus: a string (or list of char) that includes all possibile letter in the data. 
    """
    if type(corpus) is str:
        corpus = list(corpus)
    corpus_arr = np.expand_dims(corpus, 1)  # n x 1 array
    encoder = preprocessing.OneHotEncoder()
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(corpus_arr)
    encoder.fit(label_enc.transform(corpus_arr).reshape(-1, 1))
    return label_enc, encoder

def one_hot_2(seq, encoder, label_enc):
    """
    seq: a string or a list of char
    encoder: a sklearn.preprocessing.OneHotEncoder instance previously fit on a corpus. 
    """ 
    if type(seq) is str:
        seq = list(seq)
    seq_arr = np.expand_dims(seq, 1)  # n x 1 array
    oh_encoded = encoder.transform(label_enc.transform(seq_arr).reshape(-1, 1)).toarray()
    return oh_encoded

def colToMat2(x):
    """
    x: a [21, n] matrix 
    """
    return np.tile(x.transpose(), (x.shape[1],1,1))

def seq_to_volume(aa, aa_encoder, aa_label_enc, q8, q8_encoder, q8_label_enc, msa):
    """
    aa: amino acid string ("ADE") or list of char (["A", "D", "E"])
    aa_encoder: a sklearn.preprocessing.OneHotEncoder instance previously fit on the aa corpus. 
    q8: amino acid string ("ADE") or list of char (["A", "D", "E"])
    q8_encoder: a sklearn.preprocessing.OneHotEncoder instance previously fit on the q8 corpus. 
    msa: a [21, protein_length] matrix of MSA features. 
    """
    A_new = one_hot_2(aa, aa_encoder, aa_label_enc).transpose()
    Q_new = one_hot_2(q8, q8_encoder, q8_label_enc).transpose()

    msa_mat1 = colToMat2(msa)
    msa_mat2 = msa_mat1.transpose((1,0,2))
    
    aa_mat1 = colToMat2(A_new)
    aa_mat2 = aa_mat1.transpose((1,0,2))
    
    q8_mat1 = colToMat2(Q_new)
    q8_mat2 = q8_mat1.transpose((1,0,2))
    
    final_matrix = np.concatenate([msa_mat2, msa_mat1, aa_mat2, aa_mat1, q8_mat2, q8_mat1], axis=2)
#     below correspond to the previous ordering (the ordering doesn't really matter)
#     n = len(aa)
#     final_matrix = np.zeros((n,n,100))
#     final_matrix[:,:,:42:2] = msa_mat2
#     final_matrix[:,:,1:42:2] = msa_mat1
#     final_matrix[:,:,42:84:2] = aa_mat2
#     final_matrix[:,:,43:84:2] = aa_mat1
#     final_matrix[:,:,84::2] = q8_mat2
#     final_matrix[:,:,85::2] = q8_mat1
    return final_matrix

def get_patch(V, k, row, col):
    patch = V[row:(row + k + 1), col:(col + k + 1)]
    return patch

def get_sample(V, s):
    w, h = V.shape[0:2]
    w_out, h_out = w // s, h // s
    output = np.zeros_like(V)[:w_out, :h_out]
    for i in range(0, w_out):
        for j in range(0, h_out):
            output[i, j] = V[i * s, j * s]
    return output

def train_samples(v_samples, d_samples):
    """ TODO: Implement your model here. """ 
    s = v_samples[0].shape[0]
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, 
                     input_shape=v_samples[0].shape, padding='same', activation='relu'))
    #model.add(MaxPool2D(pool_size=(2,2), padding='valid')) # (n_proteins, 32, 32, 64)
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1,
                     padding='same', activation='relu'))
    #model.add(MaxPool2D(pool_size=(2,2), padding='valid')) # (n_proteins, 16, 16, 128)
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1,
                     padding='same', activation='relu'))
    #model.add(MaxPool2D(pool_size=(2,2), padding='valid')) # (n_proteins, 8, 8, 256)
    model.add(MaxPool2D(pool_size=(2,2), padding='valid')) # (n_proteins, 4, 4, 512) -> (4, 4, 256)
    model.add(Flatten())
    model.add(Dense(s * s, activation='relu'))
    model.add(Reshape((s, s)))

    print(model.summary())

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['mae'])

    n_epoch = 20
    batch_size = 1
    validate_ratio = 0.1
    early_stopping = EarlyStopping(monitor='val_loss',
                                    min_delta=0.001,
                                    patience=1,
                                    verbose=0)
    model.fit(np.array(v_samples), np.array(d_samples), batch_size=batch_size,
              epochs=n_epoch, callbacks=[early_stopping], verbose=1, 
              validation_split=validate_ratio)
    return model

def predict_sample(model, v_sample):
    return model.predict(v_sample[None, :, :, :])

def prepare_sample_and_predict(model, V, k, s, protein_length, mask, d_matrix, row, col):
    v_patch = get_patch(V, k, row, col)
    v_sample = get_sample(v_patch, s)
    d_sample = predict_sample(model, v_sample)
    for a in range(s):
        for b in range(s):
            mask[a * s + row, b * s + col] = 1
            d_matrix[a * s + row, b * s + col] = d_sample[0, a, b]
    return mask, d_matrix

def predict_and_concat_samples(model, V, k, s, protein_length):
    d_samples = []
    mask = np.zeros((protein_length, protein_length))
    d_matrix = np.zeros((protein_length, protein_length))
    for i in range(protein_length // k):
        for j in range(protein_length // k):
            row = i * k
            col = j * k
            mask, d_matrix = prepare_sample_and_predict(model, V, k, s, protein_length, mask, d_matrix, row, col)

    # Handle the edge-cases (literally) when protein_length not divisible by k.
    for i in range(protein_length // k):
        row = i * k
        col = protein_length - k
        mask, d_matrix = prepare_sample_and_predict(model, V, k, s, protein_length, mask, d_matrix, row, col)

        row = protein_length - k
        col = i * k
        mask, d_matrix = prepare_sample_and_predict(model, V, k, s, protein_length, mask, d_matrix, row, col)
    return d_matrix, mask

def downup(X):
    Xdown = imresize(X, 0.5, interp='bilinear', mode=None)
    Xdownup = imresize(Xdown, X.shape, interp='bilinear')
    return Xdownup

def diffuse(D, mask):
    X_1 = np.multiply(D, mask)
    ones = np.ones(D.shape)
    iters = 25
    for i in range(iters):
        if (i == 0): 
            X_iter = X_1
        else: 
            X_iter = np.multiply(D, mask) + np.multiply(downup(X_iter), ones - mask)
    return X_iter

# TODO: Change this if you use different/multiple folds. You may find it useful to
# load proteins on the fly in minibatches to save storage.
n_proteins = 1
k = 64 # TODO: Play around with this and s if you desire. 
s = 8

aa_label_enc, aa_enc = initialize_one_hot_encoder(AA_CORPUS)
q8_label_enc, q8_enc = initialize_one_hot_encoder(Q8_CORPUS)

v_samples = []
d_samples = []
for i in range(n_proteins):
    # Check volume representation
    volume = seq_to_volume(train_pdb_aas[i], aa_enc, aa_label_enc, train_q8s[i], q8_enc, q8_label_enc, np.array(train_msas[i]))

    d_matrix = train_dcalphas[i]

    n_patches = 15
    for j in range(n_patches):
        w, h = volume.shape[0:2]
        row = np.random.randint(0, w - k)
        col = np.random.randint(0, h - k)
        v_patch = get_patch(volume, k, row, col)
        d_patch = get_patch(d_matrix, k, row, col)
        v_sample = get_sample(v_patch, s)
        d_sample = get_sample(d_patch, s)
        v_samples.append(v_sample)
        d_samples.append(d_sample)

model = train_samples(v_samples, d_samples)

test_pdbs = []
test_pdb_aas = []
test_q8s = []
test_msas = []

test_data = load_data('data/test.pkl')
indices, pdbs, length_aas, pdb_aas, q8s, msas = test_data
set_trace()
test_pdbs += pdbs
test_pdb_aas += pdb_aas
test_q8s += q8s
test_msas += msas

test_volume = seq_to_volume(test_pdb_aas[0], aa_enc, aa_label_enc, test_q8s[0], q8_enc, q8_label_enc, np.array(test_msas[0]))
d_matrix, mask = predict_and_concat_samples(model, test_volume, k, s, len(test_pdb_aas[0]))
full_d_matrix = diffuse(d_matrix, mask)
