from torch.utils.checkpoint import checkpoint
"""
 Original paper: https://arxiv.org/pdf/1611.07004.pdf
 
 """
from config import *

'''
Generator:    
The encoder-decoder architecture consists of:
encoder:
C64-C128-C256-C512-C512-C512-C512-C512
decoder:
CD512-CD512-CD512-C512-C256-C128-C64
'''

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()



