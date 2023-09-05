# infer_grounding_dino


## :hammer_and_wrench: Install 

Exceptionally for the GroundingDINO algorithm, if you have a CUDA environment, you need to [set the environment variable CUDA_HOME](https://github.com/IDEA-Research/GroundingDINO#hammer_and_wrench-install). If CUDA is not available GroundingDINO will be compiled under CPU-only mode. 

The CUDA Toolkit is the main package that contains all the necessary tools and libraries to develop and run CUDA applications. When you install the CUDA Toolkit, it gets installed in a directory that you specify during the installation process. The default installation directory on:‍

- Windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\‍
- Linux: /usr/local/cuda/

#### Windows
To add an environment variable on Windows 10/11:
1. On the Windows taskbar, search for ‘Edit the system environment variables’
2. On the Advanced tab, click Environment VariablesClick New to create a new environment variable. 
3. Click Edit to modify an existing environment variable.
4. After creating or modifying the environment variable, click Apply and then OK to have the change take effect.

### Linux
If you want to set the CUDA_HOME permanently, store it using:
```bash
echo 'export CUDA_HOME=/path/to/cuda' >> ~/.bashrc
```
after that, source the bashrc file and check CUDA_HOME:
```bash
source ~/.bashrc
echo $CUDA_HOME
```
In this example, /path/to/cuda-11.3 should be replaced with the path where your CUDA toolkit is installed. You can find this by typing which nvcc in your terminal:

For instance, if the output is /usr/local/cuda/bin/nvcc, then
```bash
export CUDA_HOME=/usr/local/cuda
```


## :rocket: Run with Ikomia API
```python

from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display


# Init your workflow
wf = Workflow()    

# Add the Grounding DINO Object Detector
dino = wf.add_task(name="infer_grounding_dino", auto_connect=True)

dino.set_parameters({
    "prompt": "laptops . smartphone . headphone .",
    "conf_thres": "0.35",
    "conf_thres_text": "0.25"
})

# Run on your image  
# wf.run_on(path="path/to/your/image.png")
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_work.jpg")

# Inspect your results
display(dino.get_image_with_graphics())
