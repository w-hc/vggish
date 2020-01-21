'''
Replicate the tensorflow repo smoke test and confirm that the feature extractor
works as expected
'''
import numpy as np
import torch

from torchvggish.model import vggish
from torchvggish.input_process import waveform_to_input


def gen_test_input():
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


@torch.no_grad()
def test_embeddings():
    model = vggish(with_classifier=False, pretrained=True)
    model.cuda()

    x, sample_rate, expected_stats = gen_test_input()
    x = torch.from_numpy(x).reshape(1, -1).float()  # [C, L]
    # note that default torch processing differs a little bit from tf processing
    # to pass the original smoke test use the tf processing
    x = waveform_to_input(x, sample_rate, method='tf')
    x = x.cuda()

    embeddings = model(x)
    embeddings = embeddings.cpu().numpy()

    mean, std = np.mean(embeddings), np.std(embeddings)
    print('expected mean {} vs actual mean {}'.format(
        expected_stats['embedding_mean'], mean)
    )
    print('expected std {} vs actual std {}'.format(
        expected_stats['embedding_std'], std)
    )


if __name__ == '__main__':
    test_embeddings()
