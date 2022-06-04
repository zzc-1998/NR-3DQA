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
  curvature = cloud.points['curvature(11)'].to_numpy()
  anisotropy = cloud.points['anisotropy(11)'].to_numpy()
  linearity = cloud.points['linearity(11)'].to_numpy()
  planarity = cloud.points['planarity(11)'].to_numpy()
  sphericity = cloud.points['sphericity(11)'].to_numpy()


  #begin color projection
  print("Begin color feature extraction.")
  rgb_color = cloud.points[['red','green','blue']].to_numpy()/255
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


