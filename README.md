# UniTraj: earning a Universal Trajectory Foundation Model from Billion-Scale Worldwide Traces

## Overview
This repository contains the implementation of the NeurIPS 2025 accepted paper UniTraj. It is a universal trajectory foundation model designed to overcome the limitations of existing methods, such as task specificity, regional dependency, and data sensitivity. 


### Project Structure
```bash
.
├── data/                           # Data storage directory
│   └── worldtrace_sample.pkl      # WorldTrace dataset sample 
├── utils/                          # Utility modules and model components
│   ├── __init__.py                # Package initialization file
│   ├── unitraj.py                 # UniTraj model core implementation
│   ├── dataset.py                 # Dataset processing and loading, containing masking and resampling strategies
│   ├── config.py                  # Model and training configuration parameters
│   ├── logger.py                  # Colored logging system
│   └── EMA.py                     # Exponential Moving Average (EMA) helper
├── main.py                        # Main training script
├── load_see_data.ipynb           # Data loading and visualization notebook
├── requirements.txt              # Python dependency list
├── LICENSE                       # Open source license
└── README.md                     # Project documentation
```

### Requirements

This code is implemented in Python and based on the PyTorch framework. To ensure compatibility, please install the following dependencies:

#### Basic Environment
- **Python**: 3.8+
- **PyTorch**: 1.8.0+

#### Core Dependencies
- **numpy** (>=1.19.0): Numerical computation
- **pandas** (>=1.1.0): Data processing
- **matplotlib** (>=3.3.0): Data visualization
- **einops**: Simplified tensor operations
- **timm**: Vision Transformer model library
- **rdp**: Ramer-Douglas-Peucker algorithm (trajectory simplification)
- **colored**: Colored log output
- **folium**: Map visualization (for data display)


#### Running
you can run the code by running the following command:

```python

python main.py

```

## 📁 Dataset

<img src="./Logo.png" alt="Logo" style="zoom:30%;" />

The full WorldTrace dataset is released in 🤗 [Huggingface](https://huggingface.co/datasets/OpenTrace/WorldTrace) and ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAAAXNSR0IArs4c6QAAAARzQklUCAgICHwIZIgAAAH/SURBVGiB7ZkxTtxAFIa/WbFUsDJNukiL6KPkBssdaFIhGqJ0OQdHQEhZ0aQMK0UhEgW+gSlSgEJ2V1nRgEBWgMaAXwrkIijPMzbZtS3N17434//TeGZkGTwej6dKjFZ4913ENjg8tT/g5NWRS44YiDHsAQOW34QugwBaro1TJgC6CB8QDhlFh4yirsvAugj8jdBDiBhFr22t9RR4JED4bFuJOgvA42v1Ma+h7gIAPX5EPa3YBAGYY10rNUNAUDdzMwSgqxXmtML+rvUe49dWS70ICzGMbA8LtEJTVkDFC1SNF6gaL1A1jRdQL6Kd97fy4uXzJk/iNmtb89bL7tOm/evv7bb55zzqClxMDOcT27Q6SdzmLm6Xn8CR3FeorMSswoPDHigqMcvw4LiJXSVmHR4KnEI2iSrCQ8FjVJOoKjyUuAeeSlQZHnI+aPK4mBhACBarDQ8lBQCOJ0KwIKx08vtcLqmDsxtbS6wVSglcknIlwtV1AsBKZ77MNM6IiCpQeA9k4TOG1wk/fyclo7lhTGtPqxUSeBo+Y9oSqZGBVnMW0MJnTEtCIOx/WQy1upOALXzGFCRiebjfyGuwCriGz/hvEiLjVO5X+9+WxnltuQJFw2c8V0IgTNOH1f7XJevvHfUYLRs+Y1jsiI0FxgZzlKbpoL/fUU8dj8fjqRd/AB3a6/4sQ/d7AAAAAElFTkSuQmCC) [Modelscope](https://www.modelscope.cn/datasets/opentrace/WorldTrace).

We also provide a sample of the WorldTrace dataset in the *data/ directory* to help you get started quickly.

- data/worldtrace_sample.pkl: A subset of the dataset containing 1,000 trajectories.

- load_see_data.ipynb: A Jupyter Notebook that demonstrates how to load the sample data and visualize the trajectories.

## 📝 Citation
If you find our work useful in your research, please consider citing our paper:
```ini
@article{unitraj2025,
  title={UniTraj: Learning a Universal Trajectory Foundation Model from Billion-Scale Worldwide Traces},
  author={Zhu, Yuanshao and Yu, James Jianqiao and Zhao, Xiangyu and Zhou, Xun and Han, Liang and Wei, Xuetao and Liang, Yuxuan},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
```