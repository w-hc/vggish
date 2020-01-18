import os
import os.path as osp
import numpy as np
import torch
import tensorflow as tf
import tensorflow.contrib.slim as slim

from torchvggish.model import VGGish, VGGishClassify
from vggish import vggish_input, vggish_slim

ckpt_paths = {
    'orig': 'ckpt/vggish_model.ckpt',
    'andrew': 'ckpt/net-008000.tf',
}


def orig_tf_model():
    model = TF_Model(ckpt_paths['orig'], add_classifier=False)
    return model


def owens_tf_model():
    model = TF_Model(ckpt_paths['andrew'], add_classifier=True)
    return model


class TF_Model():
    def __init__(self, ckpt_fname, add_classifier=False):
        self.add_classifier = add_classifier
        num_time_samples = 3
        spec_ph = tf.placeholder(tf.float32, (num_time_samples, 96, 64))
        embeddings = vggish_slim.define_vggish_slim(spec_ph, training=True)
        self.spec_ph, self.embeddings = spec_ph, embeddings
        if add_classifier:
            with tf.variable_scope('mymodel'):
                num_units = 100
                num_classes = 527
                fc = slim.fully_connected(embeddings, num_units)
                self.logits = slim.fully_connected(
                    fc, num_classes, activation_fn=None, scope='logits'
                )

        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, ckpt_fname)

    def state_dict(self):
        state_dict = {}
        tvars = tf.trainable_variables()
        tvars_vals = self.sess.run(tvars)
        for var, val in zip(tvars, tvars_vals):
            print('{:<40} {}'.format(var.name, var.shape))
            state_dict[var.name] = val
        return state_dict

    def forward(self, x):
        to_ret = [self.embeddings, self.logits] \
                if self.add_classifier else self.embeddings
        ret = self.sess.run(
            to_ret, {self.spec_ph: x}
        )
        return ret


def gen_test_input():
    '''this is the smoke test used by Google's vggish repo'''
    # Generate a 1 kHz sine wave at 44.1 kHz (we use a high sampling rate
    # to test resampling to 16 kHz during feature extraction).
    num_secs = 3
    freq = 1000
    sample_rate = 44100
    t = np.linspace(0, num_secs, int(num_secs * sample_rate))
    x = np.sin(2 * np.pi * freq * t)
    expected_stats = {
        'embedding_mean': 0.131,
        'embedding_std': 0.238,
        'post_pca_mean': 123.0,
        'post_pca_std': 75.0
    }
    return x, sample_rate, expected_stats


def convert_orig_model():
    tf_model = orig_tf_model()
    torch_model = VGGish()
    torch_model = param_copy(tf_model, torch_model)

    data, sample_rate, expected_stats = gen_test_input()
    spec = vggish_input.waveform_to_examples(data, sample_rate)

    # test that the 2 models agree on sample input
    torch_out = torch_numpy_forward(torch_model, spec)
    tf_out = tf_model.forward(spec)
    assert np.allclose(torch_out, tf_out, atol=1e-6)
    max_diff = np.abs(torch_out - tf_out).max()
    print("max diff in model output is {}".format(max_diff))

    return torch_model


def convert_owens_model():
    tf_model = owens_tf_model()
    torch_model = VGGishClassify()
    torch_model = param_copy(tf_model, torch_model)

    data, sample_rate, expected_stats = gen_test_input()
    spec = vggish_input.waveform_to_examples(data, sample_rate)

    # test that the 2 models agree on sample input
    torch_logits = torch_numpy_forward(torch_model, spec)
    embed, logits  = tf_model.forward(spec)
    assert np.allclose(torch_logits, logits, atol=1e-6)
    max_diff = np.abs(logits - torch_logits).max()
    print("max diff in model output is {}".format(max_diff))

    return torch_model


def param_copy(tf_model, torch_model):
    tf_params_list = list(tf_model.state_dict().values())
    tf_params_list = [ to_torch_tensor(p) for p in tf_params_list ]
    torch_params_list = list(torch_model.parameters())
    assert len(tf_params_list) == len(torch_params_list)
    for torch_p, tf_p in zip(torch_params_list, tf_params_list):
        assert torch_p.shape == tf_p.shape
        torch_p.data.copy_(tf_p)
    return torch_model


@torch.no_grad()
def torch_numpy_forward(model, x):
    '''quick wrapper'''
    x = torch.from_numpy(x).unsqueeze(dim=1).float()
    x = model(x).cpu().numpy()
    return x


def to_torch_tensor(weights):
    if len(weights.shape) == 4:
        tensor = torch.from_numpy(weights.transpose(3, 2, 0, 1)).float()
    else:
        tensor = torch.from_numpy(weights.T).float()
    return tensor


def convert_and_save_orig_weights(save_root):
    model = convert_orig_model()
    path = osp.join(save_root, 'vggish_orig.pth')
    torch.save(model.state_dict(), path)


def convert_and_save_owens_weights(save_root):
    model = convert_owens_model()
    path = osp.join(save_root, 'vggish_with_classifier.pth')
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    save_root = 'converted_weights'
    os.makedirs(save_root, exist_ok=True)
    '''
    Note that my TF_Model is using the global graph; hence at each script
    invocation only a single model can be converted;
    DO NOT uncomment both lines at the same time
    '''
    convert_and_save_orig_weights(save_root)
    # convert_and_save_owens_weights(save_root)
