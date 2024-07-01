import mindspore
import numpy as np
from mindspore import Tensor, mint

tensor = Tensor(np.array([0, 1, 2]), mindspore.int32)
num_classes = 3
output = mint.nn.functional.one_hot(tensor, num_classes)
print(output)
