import numpy as np
import torch
from src.models_python.WatsonMixtureEM import Watson as W_EM
from src.models_pytorch.Watson_torch import Watson as W_t
k = torch.ones(2,dtype=torch.double)*12686
a=W_EM.kummer_log2(0,0.5,1.5,np.array(k))
b=W_t.kummer_log2(0,0.5,1.5,k)

print(a)
print(b)