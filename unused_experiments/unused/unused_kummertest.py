import numpy as np
import torch
from src.unused_models_python.WatsonMixtureEM import Watson as W_EM
from src.models_pytorch.Watson_torch import Watson as W_t
k = torch.tensor([186,-100])
a=W_EM.kummer_log(W_EM,0.5,1.5,np.array(k))
b=W_t.kummer_log(W_t,0.5,1.5,k)

print(a)
print(b)