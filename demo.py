import os
import argparse
from model.cnn_geometric_model import CNNGeometric
from data.pf_dataset import PFDataset
from data.download_datasets import download_PF_willow
from image.normalization import NormalizeImageDict, normalize_image
from geotnf.point_tnf import *
import matplotlib.pyplot as plt
from collections import OrderedDict