### convert the 2 tensorflow vggish models into pytorch
Model 1: the original model that only contains modules up to embedding layer
Model 2: Andrew's fully trained model with the extra classifier on top

use the following file hierarchy
```
scripts/
    convert_tf_to_torch_with_test.py
    vggish/ (Andrew's modified tf package)
        vggish_input.py
        vggish_slim.py
        ...
    ckpt/
        vggish_model.ckpt (Google's released ckpt)
        net-008000.tf* (Andrew's trained ckpt)
        ...
```
The Google ckpt is available [here](https://storage.googleapis.com/audioset/vggish_model.ckpt).
Invoke this conversion script from the command line
Tested with tensorflow 1.14.0
