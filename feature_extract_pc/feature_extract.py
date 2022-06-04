import numpy as np
import pandas as pd
from skimage import color
from nss_functions import * 
from pyntcloud import PyntCloud
import os 

def get_feature_vector(objpath):  
  #load colored point cloud
  print("Begin loading point cloud")
  cloud = PyntCloud.from_file(objpath)
  
  #begin geometry projection
  print("Begin geometry feature extraction.")
  k_neighbors = cloud.get_neighbors(k=10)
  ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
  cloud.add_scalar_field("curvature", ev=ev)
  cloud.add_scalar_field("anisotropy",ev=ev)
  cloud.add_scalar_field("linearity",ev=ev)
  cloud.add_scalar_field("planarity",ev=ev)
  cloud.add_scalar_field("sphericity",ev=ev)
  features = np.array(cloud.points.iloc[:,3:])
  # features
  # x y z (nx ny nz)(if have normals) r g b ... 
  # if the point cloud has normals please use
  # features = np.array(cloud.points.iloc[:,6:])
  curvature = features[:,6]
  anisotropy = features[:,7]
  linearity = features[:,8]
  planarity = features[:,9]
  sphericity = features[:,10]

  #begin color projection
  print("Begin color feature extraction.")
  rgb_color = features[:,:3]/255
  lab_color = color.rgb2lab(rgb_color)
  l = lab_color[:,0]
  a = lab_color[:,1]
  b = lab_color[:,2]
  
  
  print("Begin NSS parameters estimation.")
  # computer nss parameters
  nss_params = []
  for tmp in [l,a,b,curvature,anisotropy,linearity,planarity,sphericity]:
      params = get_nss_param(tmp)
      #flatten the feature vector
      nss_params = nss_params + [i for item in params for i in item]
  return nss_params


