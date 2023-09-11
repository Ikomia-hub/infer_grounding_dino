<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_grounding_dino/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_grounding_dino</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_grounding_dino">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_grounding_dino">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_grounding_dino/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_grounding_dino.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

The Algorithm proposes a zero-shot object grounding model that can localize objects in an image with a natural language query. 

![Grounding Dino dog detection](https://raw.githubusercontent.com/Ikomia-hub/infer_grounding_dino/main/icons/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display


# Init your workflow
wf = Workflow()    

# Add the Grounding DINO Object Detector
dino = wf.add_task(name="infer_grounding_dino", auto_connect=True)

# Run on your image  
# wf.run_on(path="path/to/your/image.png")
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_dog.png")

# Inspect your results
display(dino.get_image_with_graphics())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 'Swin-T':  The GroundingDINO algorithm has two different checkpoint models: ‘Swin-B’ and ‘Swin-T’, with respectively, 172M and 341M of parameters.  
- **prompt** (str) - default 'car . person . dog .': Text prompt for the model
- **conf_thres** (float) - default '0.35': Box threshold for the prediction‍
- **conf_thres_text** (float) - default '0.25': Text threshold for the prediction
- **cuda** (bool): If True, CUDA-based inference (GPU). If False, run on CPU

**Parameters** should be in **strings format**  when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()    

# Add the Grounding DINO Object Detector
dino = wf.add_task(name="infer_grounding_dino", auto_connect=True)

dino.set_parameters({
    "model_name": "Swin-B",
    "prompt": "laptops . smartphone . headphone .",
    "conf_thres": "0.35",
    "conf_thres_text": "0.25"
})

# Run on your image  
# wf.run_on(path="path/to/your/image.png")
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_work.jpg")

# Inspect your results
display(dino.get_image_with_graphics())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_grounding_dino", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_dog.png")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

## :fast_forward: Advanced usage 

Check out the [Grounding Dino blog post](https://www.ikomia.ai/blog/zero-shot-object-detection-with-grounding-dino-using-the-ikomia-api) for more information on this algorithm.
