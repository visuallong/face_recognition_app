from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.layers import Convolution2D, MaxPool2D, Input, Lambda, concatenate
from keras import backend as K
from keras import Sequential, Model

def vgg_face_model():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model

def deep_rank_model():
    vgg_face_model_ = vgg_face_model()
    vgg_face_weights_path = r'storage\model\vgg_face_weights.h5'
    vgg_face_model_.load_weights(vgg_face_weights_path)
    x = vgg_face_model_.layers[-3].output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2622, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(2622, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda x: K.l2_normalize(x,axis=1))(x)
    # x = Lambda(K.l2_normalize)(x)
    vgg_face_model_ = Model(inputs=vgg_face_model_.input, outputs=x)

    first_input = Input(shape=(224, 224, 3))
    first_conv = Conv2D(96, kernel_size=(8,8), strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)
    # first_max = Lambda(K.l2_normalize)(first_max)

    second_input = Input(shape=(224, 224, 3))
    second_conv = Conv2D(96, kernel_size=(8,8), strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7), strides=(4,4), padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)
    # second_max = Lambda(K.l2_normalize)(second_max)


    merge_one = concatenate([first_max, second_max])
    merge_two = concatenate([merge_one, vgg_face_model_.output])
    emb = Dense(4096)(merge_two)
    emb = Dense(128)(emb)
    l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)
    # l2_norm_final = Lambda(K.l2_normalize)(emb)
                        
    final_model = Model(inputs=[first_input, second_input, vgg_face_model_.input], outputs=l2_norm_final)

    return final_model
