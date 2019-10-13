# First file

""" From https://github.com/aqlaboratory/rgn
    Geometric TensorFlow operations for protein structure prediction.

    There are some common conventions used throughout.

    BATCH_SIZE is the size of the batch, and may vary from iteration to iteration.
    NUM_STEPS is the length of the longest sequence in the data set (not batch). It is fixed as part of the tf graph.
    NUM_DIHEDRALS is the number of dihedral angles per residue (phi, psi, omega). It is always 3.
    NUM_DIMENSIONS is a constant of nature, the number of physical spatial dimensions. It is always 3.

    In general, this is an implicit ordering of tensor dimensions that is respected throughout. It is:

        NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS, NUM_DIMENSIONS

    The only exception is when NUM_DIHEDRALS are fused into NUM_STEPS. Btw what is setting the standard is the builtin 
    interface of tensorflow.models.rnn.rnn, which expects NUM_STEPS x [BATCH_SIZE, NUM_AAS].
"""

__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

# Imports
import numpy as np
import tensorflow as tf
import collections

# Constants
NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3
BOND_LENGTHS = np.array([145.801, 152.326, 132.868], dtype='float32') / 100  # Angstrom
BOND_ANGLES  = np.array([  2.124,   1.941,   2.028], dtype='float32')

# Functions
def angularize(input_tensor, name=None):
    """ Restricts real-valued tensors to the interval [-pi, pi] by feeding them through a cosine. """

    with tf.name_scope(name, 'angularize', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
    
        return tf.multiply(np.pi, tf.cos(input_tensor + (np.pi / 2)), name=scope)

def reduce_mean_angle(weights, angles, use_complex=False, name=None):
    """ Computes the weighted mean of angles. Accepts option to compute use complex exponentials or real numbers.

        Complex number-based version is giving wrong gradients for some reason, but forward calculation is fine.

        See https://en.wikipedia.org/wiki/Mean_of_circular_quantities

    Args:
        weights: [BATCH_SIZE, NUM_ANGLES]
        angles:  [NUM_ANGLES, NUM_DIHEDRALS]

    Returns:
                 [BATCH_SIZE, NUM_DIHEDRALS]

    """

    with tf.name_scope(name, 'reduce_mean_angle', [weights, angles]) as scope:
        weights = tf.convert_to_tensor(weights, name='weights')
        angles  = tf.convert_to_tensor(angles,  name='angles')

        if use_complex:
            # use complexed-valued exponentials for calculation
            cwts =        tf.complex(weights, 0.) # cast to complex numbers
            exps = tf.exp(tf.complex(0., angles)) # convert to point on complex plane

            unit_coords = tf.matmul(cwts, exps) # take the weighted mixture of the unit circle coordinates

            return tf.angle(unit_coords, name=scope) # return angle of averaged coordinate

        else:
            # use real-numbered pairs of values
            sins = tf.sin(angles)
            coss = tf.cos(angles)

            #y_coords = tf.matmul(weights, sins)
            #x_coords = tf.matmul(weights, coss)
            y_coords = tf.tensordot(weights, sins, axes=1)
            x_coords = tf.tensordot(weights, coss, axes=1)

            return tf.atan2(y_coords, x_coords, name=scope)

def reduce_l2_norm(input_tensor, reduction_indices=None, keep_dims=None, weights=None, epsilon=1e-12, name=None):
    """ Computes the (possibly weighted) L2 norm of a tensor along the dimensions given in reduction_indices.

    Args:
        input_tensor: [..., NUM_DIMENSIONS, ...]
        weights:      [..., NUM_DIMENSIONS, ...]

    Returns:
                      [..., ...]
    """
    # sqrt(sum(x_ij**2))
    with tf.name_scope(name, 'reduce_l2_norm', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
        
        input_tensor_sq = tf.square(input_tensor)
        if weights is not None: input_tensor_sq = input_tensor_sq * weights

        return tf.sqrt(tf.maximum(tf.reduce_sum(input_tensor_sq, axis=reduction_indices, keepdims=keep_dims), epsilon), name=scope)

def reduce_l1_norm(input_tensor, reduction_indices=None, keep_dims=None, weights=None, nonnegative=True, name=None):
    """ Computes the (possibly weighted) L1 norm of a tensor along the dimensions given in reduction_indices.

    Args:
        input_tensor: [..., NUM_DIMENSIONS, ...]
        weights:      [..., NUM_DIMENSIONS, ...]

    Returns:
                      [..., ...]
    """

    with tf.name_scope(name, 'reduce_l1_norm', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
        
        if not nonnegative: input_tensor = tf.abs(input_tensor)
        if weights is not None: input_tensor = input_tensor * weights

        return tf.reduce_sum(input_tensor, axis=reduction_indices, keep_dims=keep_dims, name=scope)

def dihedral_to_point(dihedral, r=BOND_LENGTHS, theta=BOND_ANGLES, name=None):
    """ Takes triplets of dihedral angles (phi, psi, omega) and returns 3D points ready for use in
        reconstruction of coordinates. Bond lengths and angles are based on idealized averages.

    Args:
        dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    Returns:
                  [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """

    with tf.name_scope(name, 'dihedral_to_point', [dihedral]) as scope:
        dihedral = tf.convert_to_tensor(dihedral, name='dihedral') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

        num_steps  = tf.shape(dihedral)[0]
        batch_size = dihedral.get_shape().as_list()[1] # important to use get_shape() to keep batch_size fixed for performance reasons

        r_cos_theta = tf.constant(r * np.cos(np.pi - theta), name='r_cos_theta') # [NUM_DIHEDRALS]
        r_sin_theta = tf.constant(r * np.sin(np.pi - theta), name='r_sin_theta') # [NUM_DIHEDRALS]

        pt_x = tf.tile(tf.reshape(r_cos_theta, [1, 1, -1]), [num_steps, batch_size, 1], name='pt_x') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
        pt_y = tf.multiply(tf.cos(dihedral), r_sin_theta,                               name='pt_y') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
        pt_z = tf.multiply(tf.sin(dihedral), r_sin_theta,                               name='pt_z') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

        pt = tf.stack([pt_x, pt_y, pt_z])                                                       # [NUM_DIMS, NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
        pt_perm = tf.transpose(pt, perm=[1, 3, 2, 0])                                           # [NUM_STEPS, NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]
        pt_final = tf.reshape(pt_perm, [num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS], # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]
                              name=scope) 

        return pt_final

def point_to_coordinate(pt, num_fragments=6, parallel_iterations=4, swap_memory=False, name=None):
    """ Takes points from dihedral_to_point and sequentially converts them into the coordinates of a 3D structure.

        Reconstruction is done in parallel, by independently reconstructing num_fragments fragments and then 
        reconstituting the chain at the end in reverse order. The core reconstruction algorithm is NeRF, based on 
        DOI: 10.1002/jcc.20237 by Parsons et al. 2005. The parallelized version is described in XXX.

    Args:
        pt: [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    Opts:
        num_fragments: Number of fragments to reconstruct in parallel. If None, the number is chosen adaptively

    Returns:
            [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 
    """                             

    with tf.name_scope(name, 'point_to_coordinate', [pt]) as scope:
        pt = tf.convert_to_tensor(pt, name='pt')

        # compute optimal number of fragments if needed
        s = tf.shape(pt)[0] # NUM_STEPS x NUM_DIHEDRALS
        if num_fragments is None: num_fragments = tf.cast(tf.sqrt(tf.cast(s, dtype=tf.float32)), dtype=tf.int32)

        # initial three coordinates (specifically chosen to eliminate need for extraneous matmul)
        Triplet = collections.namedtuple('Triplet', 'a, b, c')
        batch_size = pt.get_shape().as_list()[1] # BATCH_SIZE
        init_mat = np.array([[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0], [-np.sqrt(2.0), 0, 0], [0, 0, 0]], dtype='float32')
        init_coords = Triplet(*[tf.reshape(tf.tile(row[np.newaxis], tf.stack([num_fragments * batch_size, 1])), 
                                           [num_fragments, batch_size, NUM_DIMENSIONS]) for row in init_mat])
                      # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
        
        # pad points to yield equal-sized fragments
        r = ((num_fragments - (s % num_fragments)) % num_fragments)          # (NUM_FRAGS x FRAG_SIZE) - (NUM_STEPS x NUM_DIHEDRALS)
        pt = tf.pad(pt, [[0, r], [0, 0], [0, 0]])                            # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        pt = tf.reshape(pt, [num_fragments, -1, batch_size, NUM_DIMENSIONS]) # [NUM_FRAGS, FRAG_SIZE,  BATCH_SIZE, NUM_DIMENSIONS]
        pt = tf.transpose(pt, perm=[1, 0, 2, 3])                             # [FRAG_SIZE, NUM_FRAGS,  BATCH_SIZE, NUM_DIMENSIONS]

        # extension function used for single atom reconstruction and whole fragment alignment
        def extend(tri, pt, multi_m):
            """
            Args:
                tri: NUM_DIHEDRALS x [NUM_FRAGS/0,         BATCH_SIZE, NUM_DIMENSIONS]
                pt:                  [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
                multi_m: bool indicating whether m (and tri) is higher rank. pt is always higher rank; what changes is what the first rank is.
            """

            bc = tf.nn.l2_normalize(tri.c - tri.b, -1, name='bc')                                        # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]        
            n = tf.nn.l2_normalize(tf.linalg.cross(tri.b - tri.a, bc), -1, name='n')                            # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]
            if multi_m: # multiple fragments, one atom at a time. 
                m = tf.transpose(tf.stack([bc, tf.linalg.cross(n, bc), n]), perm=[1, 2, 3, 0], name='m')        # [NUM_FRAGS,   BATCH_SIZE, NUM_DIMS, 3 TRANS]
            else: # single fragment, reconstructed entirely at once.
                s = tf.pad(tf.shape(pt), [[0, 1]], constant_values=3)                                    # FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS
                m = tf.transpose(tf.stack([bc, tf.linalg.cross(n, bc), n]), perm=[1, 2, 0])                     # [BATCH_SIZE, NUM_DIMS, 3 TRANS]
                m = tf.reshape(tf.tile(m, [s[0], 1, 1]), s, name='m')                                    # [FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS]
            coord = tf.add(tf.squeeze(tf.matmul(m, tf.expand_dims(pt, 3)), axis=3), tri.c, name='coord') # [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMS]
            return coord
        
        # loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially generating the coordinates for each fragment across all batches
        i = tf.constant(0)
        s_padded = tf.shape(pt)[0] # FRAG_SIZE
        coords_ta = tf.TensorArray(tf.float32, size=s_padded, tensor_array_name='coordinates_array')
                    # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
        
        def loop_extend(i, tri, coords_ta): # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
            coord = extend(tri, pt[i], True)
            return [i + 1, Triplet(tri.b, tri.c, coord), coords_ta.write(i, coord)]

        _, tris, coords_pretrans_ta = tf.while_loop(lambda i, _1, _2: i < s_padded, loop_extend, [i, init_coords, coords_ta],
                                                    parallel_iterations=parallel_iterations, swap_memory=swap_memory)
                                      # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS], 
                                      # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
        
        # loop over NUM_FRAGS in reverse order, bringing all the downstream fragments in alignment with current fragment
        coords_pretrans = tf.transpose(coords_pretrans_ta.stack(), perm=[1, 0, 2, 3]) # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        i = tf.shape(coords_pretrans)[0] # NUM_FRAGS

        def loop_trans(i, coords):
            transformed_coords = extend(Triplet(*[di[i] for di in tris]), coords, False)
            return [i - 1, tf.concat([coords_pretrans[i], transformed_coords], 0)]

        _, coords_trans = tf.while_loop(lambda i, _: i > -1, loop_trans, [i - 2, coords_pretrans[-1]],
                                        parallel_iterations=parallel_iterations, swap_memory=swap_memory)
                          # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]

        # lose last atom and pad from the front to gain an atom ([0,0,0], consistent with init_mat), to maintain correct atom ordering
        coords = tf.pad(coords_trans[:s-1], [[1, 0], [0, 0], [0, 0]], name=scope) # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

        return coords

def drmsd(u, v, weights, name=None):
    """ Computes the dRMSD of two tensors of vectors.

        Vectors are assumed to be in the third dimension. Op is done element-wise over batch.

    Args:
        u, v:    [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]
        weights: [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    Returns:
                 [BATCH_SIZE]
    """

    with tf.name_scope(name, 'dRMSD', [u, v, weights]) as scope:
        u = tf.convert_to_tensor(u, name='u')
        v = tf.convert_to_tensor(v, name='v')
        weights = tf.convert_to_tensor(weights, name='weights')

        diffs = pairwise_distance(u) - pairwise_distance(v)                                  # [NUM_STEPS, NUM_STEPS, BATCH_SIZE]
        norms = reduce_l2_norm(diffs, reduction_indices=[0, 1], weights=weights, name=scope) # [BATCH_SIZE]

        return norms

def pairwise_distance(u, name=None):
    """ Computes the pairwise distance (l2 norm) between all vectors in the tensor.

        Vectors are assumed to be in the third dimension. Op is done element-wise over batch.

    Args:
        u: [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]

    Returns:
           [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    """
    with tf.name_scope(name, 'pairwise_distance', [u]) as scope:
        u = tf.convert_to_tensor(u, name='u')
        u = tf.transpose(u, perm = [1,0,2])
        diffs = u - tf.expand_dims(u, 1)                                 # [NUM_STEPS, NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]
        norms = reduce_l2_norm(diffs, reduction_indices=[3], name=scope) # [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

        return norms

# Second file

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
# from geom_ops import *


# U-Net layers 
def conv_block(x, n_channels, droprate = 0.25):
    """ for UNet """
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x) 
    x = Dropout(droprate)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    return x 

def up_block(x, n_channels):
    """ for UNet """
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling1D(size = 2)(x)
    x = Conv1D(n_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(x)
    return x

def Conv_UNet(x, droprate=0.25):
    """ 1-D Convolutional UNet https://arxiv.org/abs/1505.04597 """

    conv0 = Conv1D(192, 3, padding = 'same', kernel_initializer = 'he_normal')(x) 

    conv1 = conv_block(conv0, 128, droprate)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = conv_block(pool1, 192, droprate)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = conv_block(pool2, 384, droprate)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = conv_block(pool3, 512, droprate)

    pool4 = MaxPooling1D(pool_size=2)(conv4)
    conv5 = conv_block(pool4, 1024, droprate)
    up5 = conv5

    up4 = up_block(up5, 512)
    up4 = concatenate([conv4,up4], axis = 2)
    up4 = conv_block(up4, 512, droprate)

    up4 = conv4

    up3 = up_block(up4, 384)
    up3 = concatenate([conv3,up3], axis = 2)
    up3 = conv_block(up3, 384, droprate)

    up2 = up_block(up3, 192)
    up2 = concatenate([conv2,up2], axis = 2)
    up2 = conv_block(up2, 192, droprate)

    up1 = up_block(up2, 128)
    up1 = concatenate([conv1,up1], axis = 2)
    up1 = conv_block(up1, 128, droprate)

    up1 = BatchNormalization()(up1)
    up1 = ReLU()(up1)

    return up1 

# some functions below are adapted from https://github.com/aqlaboratory/rgn

class DistanceMatrix(tf.keras.layers.Layer):
    """ Convert torsion angles to distance matrix 
    using differentiable geometric transformation. """
    def __init__(self):
        super(DistanceMatrix, self).__init__()
    
    def call(self, torsion_angles): 
        coordinates = torsion_angles_to_coordinates(torsion_angles)
        dist = coordinates_to_dist_matrix(coordinates)
        return dist, coordinates

class TorsionAngles(tf.keras.layers.Layer):
    """ computes torsion angles using softmax probabilities 
    and a learned alphabet of angles. (as an alternative to directly predictin angles) """
    def __init__(self, alphabet_size=50):
        super(TorsionAngles, self).__init__()
        self.alphabet = create_alphabet_mixtures(alphabet_size=alphabet_size)
    
    def call(self, probs): 
        torsion_angles = alphabet_mixtures_to_torsion_angles(probs, self.alphabet)
        return torsion_angles

def create_alphabet_mixtures(alphabet_size=50):
    """ Creates alphabet for alphabetized dihedral prediction. """
    init_range = np.pi 
    alphabet_initializer = tf.keras.initializers.RandomUniform(-init_range, init_range)
    alphabet_init = alphabet_initializer(shape=[alphabet_size, NUM_DIHEDRALS], dtype=tf.float32)
    alphabet = tf.Variable(name='alphabet', initial_value=alphabet_init, trainable=True)
    return alphabet  # [alphabet_size, NUM_DIHEDRALS]

def alphabet_mixtures_to_torsion_angles(probs, alphabet):
    """ Converts softmax probabilties + learned mixture components (alphabets) 
        into dihedral angles. 
    """
    torsion_angles = reduce_mean_angle(probs, alphabet)
    return torsion_angles  # [BATCH_SIZE, MAX_LEN, NUM_DIHEDRALS]


def torsion_angles_to_coordinates(torsion_angles, c_alpha_only=True):
    """ Converts dihedrals into full 3D structures. """
    original_shape = torsion_angles.shape
    torsion_angles = tf.transpose(torsion_angles, [1,0,2])
    # converts dihedrals to points ready for reconstruction.

    # torsion_angles: [MAX_LEN=768, BATCH_SIZE=32, NUM_DIHEDRALS=3]
    points = dihedral_to_point(torsion_angles) 
    # points: [MAX_LEN x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
             
    # converts points to final 3D coordinates.
    coordinates = point_to_coordinate(points, num_fragments=6, parallel_iterations=4) 
    # [MAX_LEN x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    if c_alpha_only:
        coordinates = coordinates[1::NUM_DIHEDRALS]  # calpha starts from 1
        # [MAX_LEN x 1, BATCH_SIZE, NUM_DIMENSIONS]
    coordinates = tf.transpose(coordinates, [1,0,2])  # do not use reshape
    return coordinates

def coordinates_to_dist_matrix(u, name=None):
    """ Computes the pairwise distance (l2 norm) between all vectors in the tensor.
        Vectors are assumed to be in the third dimension. Op is done element-wise over batch.
    Args:
        u: [MAX_LEN, BATCH_SIZE, NUM_DIMENSIONS]
    Returns:
           [BATCH_SIZE, MAX_LEN, MAX_LEN]
    """
    with tf.name_scope(name, 'pairwise_distance', [u]) as scope:
        u = tf.convert_to_tensor(u, name='u')
        u = tf.transpose(u, [1,0,2])
        
        diffs = u - tf.expand_dims(u, 1)                                 # [MAX_LEN, MAX_LEN, BATCH_SIZE, NUM_DIMENSIONS]
        norms = reduce_l2_norm(diffs, reduction_indices=[3], name=scope) # [MAX_LEN, MAX_LEN, BATCH_SIZE]
        norms = tf.transpose(norms, [2,0,1])
        return norms

def drmsd_dist_matrix(mat1, mat2, batch_seqlen, name=None):
    """
    mat1, mat2: [BATCH_SIZE, MAX_LEN, MAX_LEN]
    batch_seqlen: [BATCH_SIZE,]
    """

    weights = np.zeros(shape=mat1.shape, dtype=np.float32)
    for i, length in enumerate(batch_seqlen):
        weights[i, :length, :length] = 1

    with tf.name_scope(name, 'dRMSD', [mat1, mat2, weights]) as scope:
        mat1 = tf.convert_to_tensor(mat1, name='mat1')
        mat2 = tf.convert_to_tensor(mat2, name='mat2')
        weights = tf.convert_to_tensor(weights, name='weights')
        diffs = mat1 - mat2                      # [BATCH_SIZE, MAX_LEN, MAX_LEN]         
        #diffs = tf.transpose(diffs, [1,2,0])      # [MAX_LEN, MAX_LEN, BATCH_SIZE]
        #weights = tf.transpose(weights, [1,2,0])

        norms = reduce_l2_norm(diffs, reduction_indices=[1, 2], weights=weights, name=scope) # [BATCH_SIZE]
        drmsd = norms / batch_seqlen 
        return drmsd  # [BATCH_SIZE,]

def rmsd_torsion_angle(angles1, angles2, batch_seqlen, name=None):
    """
    angles1, angles2: [BATCH_SIZE, MAX_LEN]
    batch_seqlen: [BATCH_SIZE,]
    """

    weights = np.zeros(shape=angles1.shape, dtype=np.float32)
    for i, length in enumerate(batch_seqlen):
        weights[i, :length] = 1.0

    with tf.name_scope(name, 'RMSD_torsion', [angles1, angles2, weights]) as scope:
        angles1 = tf.convert_to_tensor(angles1, name='angles1')
        angles2 = tf.convert_to_tensor(angles2, name='angles2')
        weights = tf.convert_to_tensor(weights, name='weights')
        diffs = angles1 - angles2                      # [BATCH_SIZE, MAX_LEN]         

        norms = reduce_l2_norm(diffs, reduction_indices=[1], weights=weights, name=scope) # [BATCH_SIZE]
        drmsd = norms / tf.sqrt(batch_seqlen)
        return drmsd  # [BATCH_SIZE,]

def seq2ngrams(seqs, n = 1):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

def plot_train_val(train, val, title=None, savepath=None):
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.plot(train, c='g', label='train')
    ax.plot(val, c='b', label='val')
    ax.legend()
    if not title is None:
        ax.set_title(title)
    fig.savefig(savepath)
    plt.close()

def plot_dist_matrix(pred, gt, protein_names, lengths, scores, savepath=None):
    assert(pred.shape[0] == gt.shape[0])
    fig, axes = plt.subplots(2, pred.shape[0], figsize=(4 * pred.shape[0],8))
    for i, pname in enumerate(protein_names):
        axes[0, i].imshow(pred[i, :lengths[i], :lengths[i]])
        axes[0, i].set_title(pname + " prediction ({:.4g})".format(scores[i]))
    for i, pname in enumerate(protein_names):
        axes[1, i].imshow(gt[i, :lengths[i], :lengths[i]])
        axes[1, i].set_title(pname + " ground truth")
    plt.savefig(savepath)
    plt.close()

def rmsd_kaggle(rmsd_batch, seqlen_batch):
    """ rmsd across the entire batch """
    norm = tf.reduce_sum(tf.multiply(tf.square(rmsd_batch), seqlen_batch))
    rmsd = tf.sqrt(norm / tf.reduce_sum(seqlen_batch))
    return rmsd 

# Third file

def reduce_l22_norm(input_tensor, reduction_indices=None, keep_dims=None, weights=None, epsilon=1e-12, name=None):
    """ Computes the (possibly weighted) L2 norm of a tensor along the dimensions given in reduction_indices.

    Args:
        input_tensor: [..., NUM_DIMENSIONS, ...]
        weights:      [..., NUM_DIMENSIONS, ...]

    Returns:
                      [..., ...]
    """
    # sqrt(sum(x_ij**2))
    with tf.name_scope(name, 'reduce_l2_norm', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
        
        input_tensor_sq = tf.square(input_tensor)
        if weights is not None: input_tensor_sq = input_tensor_sq * weights

        return tf.maximum(tf.reduce_sum(input_tensor_sq, axis=reduction_indices, keepdims=keep_dims), epsilon)

def drmsd_dist_matrix2(mat1, mat2, batch_seqlen, width='full', name=None):
    """
    mat1, mat2: [BATCH_SIZE, MAX_LEN, MAX_LEN]
    batch_seqlen: [BATCH_SIZE,]
    """

    weights = np.zeros(shape=mat1.shape, dtype=np.float32)
    for i, length in enumerate(batch_seqlen):
        weights[i, :length, :length] = 1.0

    with tf.name_scope(name, 'dRMSD', [mat1, mat2, weights]) as scope:
        mat1 = tf.convert_to_tensor(mat1, name='mat1')
        mat2 = tf.convert_to_tensor(mat2, name='mat2')
        weights = tf.convert_to_tensor(weights, name='weights')
        if width!='full':
            weights = tf.compat.v1.matrix_band_part(weights, width, width)
        diffs = mat1 - mat2                      # [BATCH_SIZE, MAX_LEN, MAX_LEN]         
        #diffs = tf.transpose(diffs, [1,2,0])      # [MAX_LEN, MAX_LEN, BATCH_SIZE]
        #weights = tf.transpose(weights, [1,2,0])

        norms = reduce_l22_norm(diffs, reduction_indices=[1, 2], weights=weights, name=scope) # [BATCH_SIZE]
        drmsd = norms 
        return drmsd  # [BATCH_SIZE,]

def rmsd_torsion_angle2(angles1, angles2, batch_seqlen, name=None):
    """
    angles1, angles2: [BATCH_SIZE, MAX_LEN]
    batch_seqlen: [BATCH_SIZE,]
    """

    weights = np.zeros(shape=angles1.shape, dtype=np.float32)
    for i, length in enumerate(batch_seqlen):
        weights[i, :length] = 1.0

    with tf.name_scope(name, 'RMSD_torsion', [angles1, angles2, weights]) as scope:
        angles1 = tf.convert_to_tensor(angles1, name='angles1')
        angles2 = tf.convert_to_tensor(angles2, name='angles2')
        weights = tf.convert_to_tensor(weights, name='weights')
        diffs = angles1 - angles2                      # [BATCH_SIZE, MAX_LEN]         

        norms = reduce_l22_norm(diffs, reduction_indices=[1], weights=weights, name=scope) # [BATCH_SIZE]
        drmsd = norms
        return drmsd  # [BATCH_SIZE,]
      
def pairwise_distance_self(u, name=None):
    """ Computes the pairwise distance (l2 norm) between all vectors in the tensor.

        Vectors are assumed to be in the third dimension. Op is done element-wise over batch.

    Args:
        u: [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]

    Returns:
           [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    """
    with tf.name_scope('pairwise_distance') as scope:
        u = tf.convert_to_tensor(u, name='u')
        
        diffs = tf.expand_dims(u, 2) - tf.expand_dims(u, 1)                                 # [NUM_STEPS, NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]
        norms = reduce_l2_norm(diffs, reduction_indices=[3], name=scope) # [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

        return norms
