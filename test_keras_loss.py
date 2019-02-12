import numpy as np
from keras import objectives
from keras import backend as K

_EPSILON = K.epsilon()

def example_loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

def example_loss_np(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)

def mse_loss_tensor(y_true, y_pred):
    out = (y_true - y_pred)**2
    return K.mean(out, axis=-1)

def mse_loss_np(y_true, y_pred):
    out = (y_true - y_pred)**2
    return np.mean(out, axis=-1)

def log_loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -y_true * K.log(y_pred)
    return K.mean(out, axis=-1)

def log_loss_np(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -y_true * np.log(y_pred)
    return np.mean(out, axis=-1)

def OSL_log_loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    y_osl = y_true * y_pred
    y_osl_max = K.max(y_osl, axis=-1)
    y_osl_max = K.repeat_elements(y_osl_max, y_osl.shape[-1], 0)
    y_osl_max = K.reshape(y_osl_max, (y_osl.shape[0], y_osl.shape[-1]))
    y_osl = K.cast(K.equal(y_osl, y_osl_max), y_pred.dtype)
    y_osl = y_osl / K.sum(y_osl, axis=-1, keepdims=True)
    out = -K.stop_gradient(y_osl) * K.log(y_pred)
    return K.mean(out, axis=-1)

def OSL_log_loss_np(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    y_osl = y_true * y_pred
    y_osl_max = np.reshape(np.repeat(np.max(y_osl, axis=1),
                                           y_osl.shape[1]), (y_osl.shape[0],
                                                             y_osl.shape[1]))
    y_osl = np.equal(y_osl, y_osl_max)
    y_osl = y_osl / np.sum(y_osl, axis=1, keepdims=True)
    out = -y_osl * np.log(y_pred)
    return np.mean(out, axis=-1)

def EM_log_loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    Q = y_true * y_pred
    Z_em_train = Q / K.sum(Q, axis=-1, keepdims=True)
    out = -Z_em_train*K.log(y_pred)
    return K.mean(out, axis=-1)

def EM_log_loss_np(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    Q = y_true * y_pred
    Z_em_train = Q / Q.sum(axis=-1, keepdims=True)
    out = -Z_em_train*np.log(y_pred)
    return np.mean(out, axis=-1)

def check_loss(_shape, _loss_tensor, _loss_np):
    if _shape == '2d':
        shape = (13, 5)
    elif _shape == '3d':
        shape = (5, 6, 7)
    elif _shape == '4d':
        shape = (8, 5, 6, 7)
    elif _shape == '5d':
        shape = (9, 8, 5, 6, 7)
    print('Input shape = {}'.format(shape))

    y_a = np.random.randint(0, 2, shape)
    y_b = np.random.random(shape)

    out1 = K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))
    out2 = _loss_np(y_a, y_b)

    print('Output shape = {}'.format(out1.shape))
    assert out1.shape == out2.shape
    assert out1.shape == shape[:-1]
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))


def test_loss(tensor_loss, np_loss):
    shape_list = ['2d'] #, '3d', '4d', '5d']
    for _shape in shape_list:
        check_loss(_shape, tensor_loss, np_loss)
        print('======================')

if __name__ == '__main__':
    print('OSL')
    test_loss(OSL_log_loss_tensor, OSL_log_loss_np)
    print('Example Loss')
    test_loss(example_loss_tensor, example_loss_np)
    print('MSE')
    test_loss(mse_loss_tensor, mse_loss_np)
    print('Log-loss')
    test_loss(log_loss_tensor, log_loss_np)
    print('EM loss')
    test_loss(EM_log_loss_tensor, EM_log_loss_np)
