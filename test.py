import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['GEMINI_API_KEY'] = 'AIzaSyCfg9YzAU4DmDBqc_HGAu2gbuOSm-ayXO8'
# os.environ['OPENAI_API_KEY'] = 'sk-4_eQvEVmzUbMXbM_tpHz5WTss1p7uKwo1uvglwkGDlT3BlbkFJ6vbGXgNe2Bpa9RWtNv5U59eP1K-Fwo8L-cXaHxdXIA' 

# import os

# for name, value in os.environ.items():
#     print("{0}: {1}".format(name, value))
import gc
import torch

torch.cuda.empty_cache() 