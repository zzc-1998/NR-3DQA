# Paper
This the code for "No-Reference Quality Assessment for 3D Colored Point Cloud and Mesh Models" and it is the point cloud version.
The paper can be found here [http://arxiv.org/abs/2107.02041].

# How to start with the code?
## Environment settings
We test the code with Python 3.7 (and higher) on the Windows platform and the code may run on linux as well.

You should get the h5py, pyntcloud, skimage package by 

```
pip install h5py
pip install pyntcloud
pip install scikit-image
```

And the documentation for h5py can be found at [https://docs.h5py.org/en/stable/].

## Demo
The **demo.py** includes a demo for the feature extraction of colored point cloud and we provide with a colored point cloud sample **hhi_5.ply**.

## Database
The code is tested on the SJTU-PCQA database which can be downloaded at [https://smt.sjtu.edu.cn/].

## Experiment Update
We will provide the extracted features for SJTU-PCQA and our SVR experiment code in the future. 

**2021.11.12** We update the temporary experiment files for SJTU-PCQA, which includes the MOSs and extracted features for point clouds in the SJTU-PCQA database. The experment files are for reference only and do not represent the final results of the paper.

# Citation
If you find our work useful, please cite our work as:
```
@misc{zhang2021noreference,
      title={No-Reference Quality Assessment for 3D Colored Point Cloud and Mesh Models}, 
      author={Zicheng Zhang and Wei Sun and Xiongkuo Min and Tao Wang and Wei Lu and Guangtao Zhai},
      year={2021},
      eprint={2107.02041},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
If you have further questions, please email us through **zzc1998@sjtu.edu.cn**.
