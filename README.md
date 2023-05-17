# infer_grounding_dino


## :hammer_and_wrench: Install 

If you have a CUDA environment, please make sure the environment variable `CUDA_HOME` is set. It will be compiled under CPU-only mode if no CUDA available.

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