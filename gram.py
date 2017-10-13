# Copyright 2017 Xavier Snelgrove
import numpy as np

import os

import keras
from keras.layers import Lambda
from keras.models import Model

from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K

from scipy.ndimage import interpolation
from scipy import linalg
from scipy.optimize import minimize

from PIL import Image
import re
import string
from keras.utils import conv_utils

from enum import Enum
import tensorflow as tf

class JoinMode(Enum):
    AVERAGE = 'average'
    MAX = 'max'
    LOG_EUCLIDEAN = 'log_euclidean'
    AFFINE_INVARIANT = 'affine_invariant'
    def __str__(self):
        return self.value

def load_model(padding='valid', data_dir="model_data"):
    if padding == 'valid':
        fname = os.path.join(data_dir, "gatys_valid.h5")
    elif padding == 'same':
        fname = os.path.join(data_dir, "gatys_same.h5")
    else:
        raise ValueError("Invalid padding mode. Wanted one of ['padding', 'valid'], got: {}".format(padding))

    if not os.path.exists(fname):
        raise FileNotFoundError("Couldn't find model file at {}. Use `serialize_gatys_model.py` to create one")

    return keras.models.load_model(fname)

def PrintLayer(msg):
    return Lambda(lambda x: tf.Print(x, [x], message=msg, summarize=16))

def PrintLayerShape(msg):
    return Lambda(lambda x: tf.Print(x, [tf.shape(x)], message=msg, summarize=16))

def construct_gatys_model(padding='valid'):
    default_model = vgg19.VGG19(weights='imagenet')

    # We don't care about the actual predictions, and want to be able to handle arbitrarily
    # sized images. So let's do it!
    new_layers = []
    for i, layer in enumerate(default_model.layers[1:]):
        if isinstance(layer, keras.layers.Conv2D):
            config = layer.get_config()
            if i == 0:
                config['input_shape'] = (None, None, 3)
            config['padding'] = padding
            # ugh gatys has different layer naming
            old_name = config['name']
            m = re.match(r"block([0-9])_conv([0-9])", old_name)
            new_name = "conv{}_{}".format(m.group(1), m.group(2))
            config['name'] = new_name
            new = keras.layers.Conv2D.from_config(config)
        elif isinstance(layer, keras.layers.MaxPooling2D):
            config = layer.get_config()
            config['padding'] = padding
            #new = keras.layers.MaxPooling2D.from_config(config)
            new = keras.layers.AveragePooling2D.from_config(config)
        else:
            print("UNEXPECTED LAYER: ", layer)
            continue
        new_layers.append(new)
    model = keras.models.Sequential(layers=new_layers)
    gatys_weights = np.load("../gatys/gatys.npy", encoding='latin1').item() # encoding because of python2
    # Previously, we loaded weights from Keras' VGG-16. Now, instead, we'll use Gatys' VGG-19!
    for i, new_layer in enumerate(model.layers):
        if 'conv' in new_layer.name:
            layer_weights = gatys_weights[new_layer.name]
            w = layer_weights['weights']
            b = layer_weights['biases']
            new_layer.set_weights([w, b])
    model._padding_mode = padding
    return model

colour_offsets = np.asarray([103.939, 116.779, 123.68])

def preprocess(img):
    if hasattr(img, 'shape'):
        # Already arrayed and batched
        return vgg19.preprocess_input(img.copy())
    else:
        img = img_to_array(img).copy()
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

def deprocess(x):
    x = x.copy()
    x[...,:] += colour_offsets
    return x[...,::-1].clip(0, 255).astype('uint8')

def gram_node(x):
    # Modified from Keras
    assert K.image_data_format() == 'channels_last'
    if K.ndim(x) == 4:
        shape = K.shape(x)
        
        features = K.reshape(K.permute_dimensions(x, (0,3,1,2)), (shape[0], shape[3], shape[2]*shape[1]))
        # batch x channels x pixels
        features = K.permute_dimensions(features, (0, 2, 1))
        # batch x pixels x channels
    elif K.ndim(x) == 2:
        #print("NDIM 2")
        # channels x pixels
        features = K.expand_dims(x, axis=0)
    else:
        #print("NDIM 3")
        assert K.ndim(x) == 3
        # batch x channels x pixels
        
    features_shape = K.shape(features)
    #features_shape = K.print_tensor(features_shape, "Feature Shape")
        
    # features is now (batch_length, pixels, channels)
    # want gram to be (batch_length, channels, channels)
    # This gives the correlation between features within the same image
    gram = K.batch_dot(K.permute_dimensions(features, (0, 2, 1)), features)

    # Normalize the gram matrix by the number of pixels, and the number of channels to make it
    # size and layer agnostic.
    gram = gram / K.cast(features_shape[1], 'float32') / K.cast(features_shape[2], 'float32')
    #gram = gram / features_shape[1] / features_shape[2]
    return gram

def gram_layer():
    def gram_shape(input_shape):
        assert len(input_shape) == 4
        return (input_shape[0], input_shape[3], input_shape[3])
    return Lambda(gram_node, gram_shape)

def reduce_layer(a=0.4, padding_mode='valid'):
    # A 5-tap Gaussian pyramid generating kernel from Burt & Adelson 1983.
    kernel_1d = [0.25 - a/2, 0.25, a, 0.25, 0.25 - a/2]
    
    #kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # This doesn't seem very computationally bright; but there you have it.
    #kernel_4d = np.zeros((5, 5, 3, 3), 'float32')
    #kernel_4d[:,:,0,0] = kernel_2d
    #kernel_4d[:,:,1,1] = kernel_2d
    #kernel_4d[:,:,2,2] = kernel_2d
    kernel_3d = np.zeros((5, 1, 3, 3), 'float32')
    kernel_3d[:, 0, 0, 0] = kernel_1d
    kernel_3d[:, 0, 1, 1] = kernel_1d
    kernel_3d[:, 0, 2, 2] = kernel_1d

    def fn(x):
        return K.conv2d(K.conv2d(x, kernel_3d, strides=(2,1)),
                K.permute_dimensions(kernel_3d, (1, 0, 2, 3)),
                strides = (1, 2))
    
    def shape(input_shape):
        assert len(input_shape) == 4
        assert K.image_data_format() == 'channels_last'
        space = input_shape[1:-1]
        new_space = []
        for i, dim in enumerate(space):
            new_dim = conv_utils.conv_output_length(
                dim,
                5,
                padding=padding_mode,
                stride=2)
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (input_shape[3],)
    
    return Lambda(fn, shape)

def expand_layer(a=0.4, padding_mode='same'):
    kernel_1d = [0.25 - a/2, 0.25, a, 0.25, 0.25 - a/2]

    kernel_3d = np.zeros((5, 1, 3, 3), 'float32')
    kernel_3d[:, 0, 0, 0] = kernel_1d
    kernel_3d[:, 0, 1, 1] = kernel_1d
    kernel_3d[:, 0, 2, 2] = kernel_1d



    def fn(x):
        #conv_even = K.conv2d(K.conv2d(x, even_kernel_3d),
                    #K.permute_dimensions(even_kernel_3d, (1, 0, 2, 3)))
        #conv_odd = K.conv2d(K.conv2d(x, odd_kernel_3d),
                    #K.permute_dimensions(odd_kernel_3d, (1, 0, 2, 3)))
        input_shape = K.shape(x)
        
        dim1 = conv_utils.conv_input_length(
                input_shape[1],
                5,
                padding=padding_mode,
                stride=2)
        dim2 = conv_utils.conv_input_length(
                input_shape[2],
                5,
                padding=padding_mode,
                stride=2)
        
        output_shape_a = (input_shape[0], dim1, input_shape[2], input_shape[3])
        output_shape_b = (input_shape[0], dim1, dim2, input_shape[3])

        upconvolved = K.conv2d_transpose(x,
                                         kernel_3d,
                                         output_shape_a,
                                        strides = (2,1),
                                        padding=padding_mode)
        upconvolved = K.conv2d_transpose(upconvolved,
                                         K.permute_dimensions(kernel_3d, (1, 0, 2, 3)),
                                         output_shape_b,
                                        strides = (1,2),
                                        padding=padding_mode)

        return 4 * upconvolved

    
    return Lambda(fn)

def compute_gram(x):
    return gram_node(K.variable(x)).eval()

def make_gram_model(base_model):
    ''' Take an (ideally VGG-19) model, and hook up and outlet to every
    layer that outputs the gram matrix of that layer '''
    image_input = keras.layers.Input((None, None, 3))
    grams = []
    current_out = image_input
    for layer in base_model.layers:
        current_out = layer(current_out)
        grams.append(gram_layer()(current_out))
    return Model(inputs = image_input, outputs = grams, name="grams")

def make_pyramid_model(num_octaves, padding_mode='valid'):
    ''' Creates a model that blurs and halves an image num_octaves times '''
    image_input = keras.layers.Input((None, None, 3))
    
    # build a series of rescaling paths in the model
    
    gaussian_pyramid = [image_input]
    for _ in range(num_octaves-1):
        level = reduce_layer(padding_mode=padding_mode)(gaussian_pyramid[-1])
        gaussian_pyramid.append(level)
    return Model(inputs = image_input, outputs = gaussian_pyramid, name="pyramid")

def make_pyramid_gram_model(pyramid_model, layer_indices, padding_mode = 'valid', data_dir="model_data"):
    base_model = load_model(padding_mode, data_dir=data_dir)
    gram_model = make_gram_model(base_model)
    pyramid_grams = []
    for output in pyramid_model.outputs:
        pyramid_grams.extend(gram_model(output))

    # we only want to keep some layers
    selected_outputs = []
    for i, output in enumerate(pyramid_grams):
        layer_i = i%len(gram_model.outputs)
        if layer_i in layer_indices:
            selected_outputs.append(output)
    return Model(inputs = pyramid_model.input, outputs = selected_outputs)

def grams_for_pyramid(pyramid_model, layer_indices, data_dir):
    base_model = load_model(padding_mode, data_dir=data_dir)
    gram_model = make_gram_model(base_model)

    pyramid_grams = []
    for output in pyramid_model.outputs:
        pyramid_grams.extend(gram_model(output))

    # we only want to keep some layers
    selected_outputs = []
    for i, output in enumerate(pyramid_grams):
        layer_i = i%len(gram_model.outputs)
        if layer_i in layer_indices:
            selected_outputs.append(output)

    return base_model.selected_outputs


def image_files_from_sources(image_sources):
    '''Sources are either image files, or directories of image files. This is not recursive,
    so you cannot nest directories '''
    image_files = []
    for file_or_directory in image_sources:
        if os.path.isdir(file_or_directory):
            image_files.extend([os.path.join(file_or_directory, f)
                    for f in os.listdir(file_or_directory) if os.path.splitext(f.lower())[-1] in ['.jpg', '.png']])
        else:
            image_files.append(file_or_directory)
    return image_files

def get_images(image_files, source_width=None, source_scale=None):
    for image_file in image_files:
        im = load_img(image_file)
        if source_width:
            im = im.resize((source_width, source_width * im.size[1] // im.size[0]), Image.LANCZOS)
        elif source_scale:
            im = im.resize((int(im.size[0] * source_scale), int(im.size[1] * source_scale)), Image.LANCZOS)
            
        prepped = preprocess(im)
        del im
        yield prepped

def get_gram_matrices_for_images(pyramid_gram_model, image_sources, source_width = None, source_scale = None, join_mode = JoinMode.AVERAGE):
    
    target_grams = []
    print("Loading image files")
    image_files = image_files_from_sources(image_sources)
    for i, prepped in enumerate(get_images(image_files, source_width=source_width, source_scale=source_scale)):
        print("{} / {}...".format(i+1, len(image_files)))
        this_grams = pyramid_gram_model.predict(prepped)
        print("got the grams!")

        if join_mode in {JoinMode.AFFINE_INVARIANT, JoinMode.LOG_EUCLIDEAN}:
            # Add a small epsilon to the diagonals to ensure a positive definite matrix
            eps = 0.05
            this_grams = [g + np.identity(g.shape[0])*eps for g in this_grams]
        if join_mode == JoinMode.AFFINE_INVARIANT:
            target_grams.append(this_grams)
        else:
            if len(target_grams) == 0:
                if join_mode == JoinMode.LOG_EUCLIDEAN:
                    target_grams = [linalg.logm(gram[0]) for gram in this_grams]
                else:
                    target_grams = this_grams
            else:
                for target_gram, this_gram in zip(target_grams, this_grams):
                    # There are likely more interesting ways to join gram matrices, including
                    # having "don't care" regions where it's not necessary to match at all. This would
                    # probably allow fusion between disparate image types to work better.
                    if join_mode == JoinMode.AVERAGE:
                        target_gram += this_gram
                    elif join_mode == JoinMode.MAX:
                        np.maximum(target_gram, this_gram, out=target_gram)
                    elif join_mode == JoinMode.LOG_EUCLIDEAN:
                        print(this_gram.shape)
                        target_gram += linalg.logm(this_gram[0])
                    else:
                        assert False
        
    # Normalize the targets
    if join_mode in {JoinMode.AVERAGE, JoinMode.LOG_EUCLIDEAN}:
        for i, target_gram in enumerate(target_grams):
            target_gram /= len(image_files)
            if join_mode == JoinMode.LOG_EUCLIDEAN:
                target_gram = linalg.expm(target_gram)
                target_gram = np.expand_dims(target_gram, -1)
                target_grams[i] = target_gram
    elif join_mode == JoinMode.AFFINE_INVARIANT:
        if len(target_grams) != 2:
            print("WARNING! affine_invariant join mode requires 2 source images")
        source_grams = target_grams # This was mis-named
        target_grams = []
        for A, B in zip(source_grams[0], source_grams[1]):
            print("A SHAPE", A.shape)
            print("B SHAPE", B.shape)
            if len(A.shape) > 2: A = A[0]
            if len(B.shape) > 2: B = B[0] # TODO: Fix, yo
            rootA = linalg.fractional_matrix_power(A, 0.5)
            rootAinv = linalg.fractional_matrix_power(A, -0.5) # hmm... non-invertible because 0 determinant?
            
            internal = 0.5 * rootAinv.dot(B).dot(rootAinv)

            interpolated = rootA.dot(linalg.expm(internal)).dot(rootA)
            target_grams.append(np.expand_dims(interpolated, -1))

    return target_grams
        
def diff_loss(model, targets):
    diff_layers = []
    for base, target in zip(model.outputs, targets):
        # TODO: I highly doubt this can handle batches properly... sums across all of them
        # Note the slight hack in the lambda with default parameter below; this allows us to effectively create
        # a closure capturing the value of target_gram rather than having them all target the *same* gram
        # (which, incidentally,creates some pretty interesting effects)
        # TODO: Remove the "prod" again after experiment
        diff_layers.append(Lambda(lambda x, target=target:K.sum(K.square(target - x[0])),
                                               output_shape = lambda input_shape: [1])([base]))
    if len(diff_layers) > 1:
        total_diff = keras.layers.add(diff_layers)
    else:
        total_diff = diff_layers[0]

    return total_diff


def cropped_diff(x):
    ''' Crop a to the shape of b'''
    a, b = x
    b_shape = K.shape(b)
    return a[:, :b_shape[1], :b_shape[2], :] - b


def laplacian_from_gaussian(pyramid_model):
    laplacian_levels = []
    for output_a, output_b in zip(pyramid_model.outputs, pyramid_model.outputs[1:]):
        expanded = expand_layer()(output_b)
        delta = Lambda(cropped_diff)([output_a, expanded])
        laplacian_levels.append(delta)
    return laplacian_levels

def lap1_diff(laplacian, frame_step=1):
    ''' Model which takes the lap-1 distance between frames `frame_step` apart
    in the batch '''
    deltas = []
    for i, lap_level in enumerate(laplacian):
        # Take the difference of the Laplacian pyramid of this layer vs. the next
        diff = Lambda(lambda lap_level, frame_step=frame_step:
                K.batch_flatten(
                    lap_level - K.concatenate([lap_level[frame_step:], lap_level[0:frame_step]], axis=0)))(lap_level)
        # scale for good measure
        diff = Lambda(lambda x, scale = 2.**-(i-1): scale*x)(diff)
        #diff = K.batch_flatten(lap_layer - K.concatenate([lap_layer[frame_step:], lap_layer[0:frame_step]], axis=0))
        deltas.append(diff) # diff: (frames, lap-pixels)

    out = keras.layers.concatenate(deltas, axis=1) # (frames, lap-pixels)
    # I use mean here instead of sum to make it more agnostic to total pixel count.
    out = Lambda(lambda x: K.mean(K.abs(x), axis=1))(out) # (frames,)
    return out
    


def lap_loss(pyramid_model, target_distance=1., order=2):
    # The pyramid model is a Gaussian pyramid, now compute the Laplacian pyramid.
    laplacian = laplacian_from_gaussian(pyramid_model)

    order_errors = []
    for frame_step in range(1,order+1):
        out = lap1_diff(laplacian, frame_step)
        out = PrintLayer("mean abs diff")(out)
        # Previously I took the square root of this mean...
        out = Lambda(lambda x: K.expand_dims(K.mean(K.square(x - target_distance*frame_step))))(out)
        order_errors.append(out)

    return keras.layers.add(order_errors)

def novelty_loss(grams, mul=1.0):
    dets = []
    for gram in grams:
        # gram will be something like (5, 64, 64)
        flat = keras.layers.Flatten()(gram)

        # ~ (5, 4096)
        covar = Lambda(lambda x: K.dot(x,K.transpose(x)),
                output_shape = lambda input_shape: [input_shape[0], input_shape[0]])(flat)
        covar = PrintLayer("covar")(covar)

        # ~ (5, 5)
        #det = Lambda(lambda x: -tf.matrix_determinant(x),
                #output_shape = lambda input_shape: [1])(covar)
        #det = Lambda(lambda x: -2*tf.reduce_sum(tf.log(tf.diag(tf.cholesky(x)))),
               #output_shape = lambda input_shape: [1])(covar)
        
        def eye_diff(x):
            shape = K.shape(x)
            return x - mul * tf.eye(shape[0], shape[1])

        det = Lambda(lambda x: K.sum(K.square(eye_diff(x))),
                output_shape = lambda input_shape: [1])(covar)
        det = PrintLayer("det")(det)
        dets.append(det)

    if len(dets) > 1:
        return keras.layers.add(dets)
    else:
        return dets[0]

def internal_novelty_loss(grams, mul=1.0):
    gram = keras.layers.Concatenate(axis=0)(grams)
    # gram will be something like (5, 64, 64)
    flat = keras.layers.Flatten()(gram)
    flat = PrintLayerShape("flat shape")(flat)

    # ~ (5, 4096)
    covar = Lambda(lambda x: K.dot(x,K.transpose(x)),
            output_shape = lambda input_shape: [input_shape[0], input_shape[0]])(flat)
    covar = PrintLayer("covar")(covar)

    # ~ (5, 5)
    #det = Lambda(lambda x: -tf.matrix_determinant(x),
            #output_shape = lambda input_shape: [1])(covar)
    #det = Lambda(lambda x: -2*tf.reduce_sum(tf.log(tf.diag(tf.cholesky(x)))),
           #output_shape = lambda input_shape: [1])(covar)
    
    def eye_diff(x):
        shape = K.shape(x)
        return x - mul * tf.eye(shape[0], shape[1])

    det = Lambda(lambda x: K.sum(K.square(eye_diff(x))),
            output_shape = lambda input_shape: [1])(covar)
    det = PrintLayer("det")(det)
    return det



def integer_interframe_distance(pyramid_model, image, shift):
    ''' How much is the lap1 diff if we shift this image by "shift" pixels?'''
    rolled_1 = np.roll(image, shift, axis=1)
    rolled_2 = np.roll(rolled_1, shift, axis=2)
    stacked = np.concatenate([image, rolled_1, rolled_2], axis=0)

    laplacian_levels = laplacian_from_gaussian(pyramid_model)
    diff = lap1_diff(laplacian_levels)
    diff_model = Model(inputs=pyramid_model.input, outputs=diff)
    
    predicted_diffs = diff_model.predict(stacked)

    print(predicted_diffs)
    return np.mean(predicted_diffs[:2]) # Ignore the third one, which is a double-shift.

def interframe_distance(pyramid_model, image, shift):
    floored = int(shift)
    a = integer_interframe_distance(pyramid_model, image, floored)
    if floored == shift:
        return a
    else:
        b = integer_interframe_distance(pyramid_model, image, floored+1)
        t = shift-floored
        return a * (1 - t) + b * (t)


def gram_loss_callable(gram_model, target_grams, shape):
    ''' Returns a function which takes in an image and outputs both the gram-matrix
    loss of that image relative to the targets, and the gradients of that loss with respect
    to the image pixels'''
    loss = diff_loss(gram_model, target_grams)

    gradients = K.gradients(loss, gram_model.input)
    if keras.backend.backend() == 'tensorflow':
        gradients = gradients[0] # This is a Keras inconsistency between theano and tf backends
    
    loss_and_gradients = K.function([gram_model.input], [loss, gradients])
    
    def callable(x):
        deflattened = x.reshape([-1] + list(shape) + [3])

        loss, grad = loss_and_gradients([deflattened])
        
        #print(formatter.format("{:q} ", float(loss)), end=' | ', flush=True)
        return loss.astype('float64'), np.ravel(grad.astype('float64'))
    return callable

def loss_and_gradients_callable(loss_model, shape):

    loss = loss_model.output
    gradients = K.gradients(loss, loss_model.input)

    if keras.backend.backend() == 'tensorflow':
        gradients = gradients[0] # This is a Keras inconsistency between theano and tf backends
    
    loss_and_gradients = K.function([loss_model.input], [loss, gradients])

    def callable(x):
        deflattened = x.reshape([-1] + list(shape) + [3])

        loss, grad = loss_and_gradients([deflattened])
        
        #print(formatter.format("{:q} ", float(loss)), end=' | ', flush=True)
        return loss.astype('float64'), np.ravel(grad.astype('float64'))
    return callable


def make_progress_callback(shape, output_directory, save_every=2):
    i = [0] # Really? Weird scope rules.
    def progress_callback(x):
        if i[0]%save_every == 0:
            channels_last = x.reshape(-1,3)
            print("\n+++ SAVING ITER {} ++++".format(i[0]))
            
            def mat_string(m):
                return " ".join(["{:.2f}".format(float(mm)) for mm in np.ravel(m)])
            reshaped = x.reshape([-1] + list(shape) + [3])
            
            deprepped = deprocess(reshaped)
            for frame_i in range(deprepped.shape[0]):
                Image.fromarray(deprepped[frame_i])\
                        .save(os.path.join(output_directory, "I{:04d}_F{:04d}.png".format(i[0], frame_i)))
        i[0] += 1
    return progress_callback




def synthesize_novelty(gram_model, width, height, x0, frame_count=1, mul=1.0, output_directory="outputs",
        save_every=10, max_iter=500, tol=1e-9, octave_step=1, internal=False):
    generated_shape = (height, width)
    
    x0_deprepped = deprocess(x0.reshape([-1] + list(generated_shape) + [3]))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for frame_i in range(x0_deprepped.shape[0]):
        Image.fromarray(x0_deprepped[frame_i])\
                .save(os.path.join(output_directory, "Aseed_F{:04d}.png".format(frame_i)))

    print("gram model outputs:", len(gram_model.outputs))
    
    # I should now have a Gram matrix for each frame.
    if internal:
        print("Internal")
        novelty = internal_novelty_loss(gram_model.outputs[::octave_step], mul=mul)
    else:
        novelty = novelty_loss(gram_model.outputs[::octave_step], mul=mul)

    loss_model = Model(inputs=gram_model.input, outputs=[novelty])
    
    optimize_me = loss_and_gradients_callable(loss_model, generated_shape)

    #optimize_me = gram_loss_callable(gram_model, target_grams, generated_shape)
    print("Generated callable")

    print("About to start minimizing...", flush=True)
    result = minimize(optimize_me, np.ravel(x0), jac=True, method="l-bfgs-b",
                      callback=make_progress_callback(generated_shape, output_directory, save_every=save_every),
                      tol=tol,
                      #bounds=bounds,
                      options={'disp': True, 'maxiter': max_iter})
    return result
                
    

def synthesize_animation(pyramid_model, gram_model, target_grams,
        width, height, frame_count=1,
        interframe_loss_weight = 1., interframe_order=2, target_interframe_distance = 50.,
        output_directory = "outputs",
        x0=None, max_iter=200, save_every=2, tol=1e-9):
    from scipy import ndimage
    from PIL import ImageFilter
    generated_shape = (height, width)

    if x0 is None:
        # Seed the optimization with random Gaussian noise (scaled by 2).
        # There are lots of interesting effects to be had by messing with this
        # initialization
        x0 = np.random.randn(*([frame_count] + list(generated_shape) + [3])) * 2

    x0_deprepped = deprocess(x0.reshape([-1] + list(generated_shape) + [3]))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for frame_i in range(x0_deprepped.shape[0]):
        Image.fromarray(x0_deprepped[frame_i])\
                .save(os.path.join(output_directory, "Aseed_F{:04d}.png".format(frame_i)))

    
    print("Generating callable...")

    print("gram model outputs:", len(gram_model.outputs))
    style_loss = diff_loss(gram_model, target_grams)
    style_loss = PrintLayer("Style Loss")(style_loss)
    if frame_count > 1:
        interframe_loss = lap_loss(pyramid_model, target_distance=target_interframe_distance, order=interframe_order)
        interframe_loss = PrintLayer("Interframe Loss")(interframe_loss)

        total_loss = keras.layers.add([style_loss, Lambda(lambda x: interframe_loss_weight*x)(interframe_loss)])
    else:
        total_loss = style_loss

    print(total_loss)
    import pdb
    loss_model = Model(inputs=pyramid_model.input, outputs=[total_loss])
    
    optimize_me = loss_and_gradients_callable(loss_model, generated_shape)

    #optimize_me = gram_loss_callable(gram_model, target_grams, generated_shape)
    print("Generated callable")

    # Could use this
    bounds = [[- colour_offsets[0], 255 - colour_offsets[0]],
              [- colour_offsets[1], 255 - colour_offsets[1]],
              [- colour_offsets[2], 255 - colour_offsets[2]]] * (x0.size//3)

    print("About to start minimizing...", flush=True)
    result = minimize(optimize_me, np.ravel(x0), jac=True, method="l-bfgs-b",
                      callback=make_progress_callback(generated_shape, output_directory, save_every=save_every),
                      tol=tol,
                      #bounds=bounds,
                      options={'disp': True, 'maxiter': max_iter})
    return result
