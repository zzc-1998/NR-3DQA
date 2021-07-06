# Paper
This the code for "No-Reference Quality Assessment for Colored Point Cloud and Mesh Based on Natural Scene Statistics" and it is the point cloud version.
The paper can be found here [http://arxiv.org/abs/2107.02041].

# How to start with the code?
## Environment settings
We test the code with Python 3.7 (and higher) on the Windows platform and the code may run on linux as well.

You should get the h5py package by 

```
pip install h5py
```

And the documentation for h5py can be found at [https://docs.h5py.org/en/stable/].

## Demo
The **feature_extract.py** includes a demo for the feature extraction of colored point cloud and we provide with a colored point cloud sample **hhi_5.ply**.

## Database
The code is tested on the SJTU-PCQA database which can be downloaded at [https://smt.sjtu.edu.cn/].

## Experiment
We will provide the extracted features for SJTU-PCQA and our SVR experiment code in the future. 

# Citation
If you find our work useful, please cite our work as:
```
@misc{zhang2021noreference,
      title={No-Reference Quality Assessment for Colored Point Cloud and Mesh Based on Natural Scene Statistics}, 
      author={Zicheng Zhang},
      year={2021},
      eprint={2107.02041},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
