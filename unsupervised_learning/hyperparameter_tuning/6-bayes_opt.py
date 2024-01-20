import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization


def preprocess_data(X, Y):
    """ Function to preprocess data"""
    X = K.applications.densenet.preprocess_input(X)
    Y = K.utils.to_categorical(Y)
    return X, Y


data = np.load('cifar10_data.npz')
x_train, y_train = preprocess_data(data['train_images'], data['train_labels'])
x_test, y_test = preprocess_data(data['test_images'], data['test_labels'])


# Pretrained DenseNet201 as base model
# Extract features from pre-trained model for train and validation sets
# Freeze the layers of the base model


input_tensor = K.Input(shape=(32, 32, 3))
resized_images = K.layers.Lambda(
    lambda image: tf.image.resize(image, (224, 224)))(input_tensor)
base_model = K.applications.DenseNet201(include_top=False, weights='imagenet',
                                        input_tensor=resized_images, input_shape=(
                                            224, 224, 3),
                                        pooling='max', classes=1000)
output = base_model.layers[-1].output
base_model = K.models.Model(inputs=input_tensor, outputs=output)


# Generate features from base model for train and validation sets
# These features will be used as input for the customized classifier

train_datagen = K.preprocessing.image.ImageDataGenerator()
train_generator = train_datagen.flow(
    x_train, y_train, batch_size=32, shuffle=False)
features_train = base_model.predict(train_generator)

val_datagen = K.preprocessing.image.ImageDataGenerator()
val_generator = val_datagen.flow(x_test, y_test, batch_size=32, shuffle=False)
features_valid = base_model.predict(val_generator)


def build_model(units=256, learning_rate=1e-4, l2=1e-2, activation=2, rate=0.5):
    initializer = K.initializers.he_normal()
    input_tensor = K.Input(shape=features_train.shape[1])
    activation_dict = {1: 'relu', 2: 'elu', 3: 'tanh'}
    layer = K.layers.Dense(units=units, activation=activation_dict[activation],
                           kernel_initializer=initializer, kernel_regularizer=K.regularizers.l2(l2=l2))
    output = layer(input_tensor)
    dropout = K.layers.Dropout(rate)
    output = dropout(output)
    softmax = K.layers.Dense(units=10, activation='softmax',
                             kernel_initializer=initializer, kernel_regularizer=K.regularizers.l2(l2=l2))
    output = softmax(output)
    model = K.models.Model(inputs=input_tensor, outputs=output)

    model.compile(optimizer=K.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    lr_reduce = K.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=2,
                                              verbose=1, mode='max', min_lr=1e-7)
    early_stop = K.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, verbose=1, mode='max')
    checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5', monitor='val_accuracy', verbose=1,
                                             save_weights_only=False, save_best_only=True, mode='max', save_freq='epoch')
    return model, lr_reduce, early_stop, checkpoint


def fit_model(model, lr_reduce, early_stop, checkpoint):
    history = model.fit(features_train, y_train, batch_size=32, epochs=20, verbose=0,
                        callbacks=[lr_reduce, early_stop, checkpoint], validation_data=(
                            features_valid, y_test),
                        shuffle=True)
    return history


def evaluate_model(model):
    evaluation = model.evaluate(features_valid, y_test)
    return evaluation


kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)

# Define the hyperparameter search space for Bayesian Optimization
bounds = [{'name': 'units', 'type': 'discrete', 'domain': (64, 128, 256, 512)},
          {'name': 'learning_rate', 'type': 'discrete',
              'domain': (1e-3, 1e-4, 1e-5, 1e-6)},
          {'name': 'l2', 'type': 'discrete', 'domain': (1e-1, 1e-2, 1e-3)},
          {'name': 'activation', 'type': 'discrete', 'domain': (1, 2, 3)},
          {'name': 'rate', 'type': 'discrete', 'domain': (0.3, 0.5, 0.7)}]


def f(x):
    try:
        previous_best_model = K.models.load_model('cifar10_best.h5')
        previous_evaluation = evaluate_model(previous_best_model)
    except Exception:
        previous_best_model = None
    model, lr_reduce, early_stop, checkpoint = build_model(units=int(x[:, 0]), learning_rate=float(x[:, 1]),
                                                           l2=float(x[:, 2]), activation=int(x[:, 3]),
                                                           rate=float(x[:, 4]))
    history = fit_model(model, lr_reduce, early_stop, checkpoint)
    evaluation = evaluate_model(model)

    if not previous_best_model:
        K.models.save_model(model, 'cifar10_best.h5',
                            overwrite=False, include_optimizer=True)
    if previous_best_model and evaluation[1] > previous_evaluation[1]:
        K.models.save_model(model, 'cifar10_best.h5',
                            overwrite=True, include_optimizer=True)

    del model, previous_best_model
    K.backend.clear_session()
    return evaluation[1]


optimizer = BayesianOptimization(f=f, domain=bounds, model_type='GP', kernel=kernel, acquisition_type='EI',
                                 acquisition_jitter=0.01, exact_feval=False, normalize_Y=False, maximize=True,
                                 verbosity=False)
optimizer.run_optimization(max_iter=30, verbosity=False)
optimizer.save_report('bayes_opt.txt')

activation_dict = {1: 'relu', 2: 'elu', 3: 'tanh'}
print("""
Optimized Parameters:
\t{0}:\t{1}
\t{2}:\t{3}
\t{4}:\t{5}
\t{6}:\t{7}
\t{8}:\t{9}
""".format(bounds[0]["name"], optimizer.x_opt[0],
           bounds[1]["name"], optimizer.x_opt[1],
           bounds[2]["name"], optimizer.x_opt[2],
           bounds[3]["name"], activation_dict[optimizer.x_opt[3]],
           bounds[4]["name"], optimizer.x_opt[4]))
print("Optimized accuracy: {0}".format(abs(optimizer.fx_opt)))

best_model = K.models.load_model('cifar10_best.h5')
loss, acc = best_model.evaluate(features_valid, y_test)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

data_path = 'bayes_opt.txt'
with open(data_path, 'r') as f:
    lines = f.read().split('\n')
for line in lines:
    print(line)
