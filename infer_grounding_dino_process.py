# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess, utils
from infer_grounding_dino.GroundingDINO.groundingdino.util.inference import load_model, predict
import infer_grounding_dino.GroundingDINO.groundingdino.datasets.transforms as T
from PIL import Image
from torchvision.ops import box_convert
import numpy as np
import os
import torch
import urllib.request


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferGroundingDinoParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.prompt = "car . person . dog ."
        self.model_name = "Swin-T"
        self.conf_thres = 0.35
        self.conf_thres_text = 0.25
        self.cuda = torch.cuda.is_available()
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = params["model_name"]
        self.prompt = params["prompt"]
        self.conf_thres = float(params["conf_thres"])
        self.conf_thres_text = float(params["conf_thres_text"])
        self.cuda = utils.strtobool(params["cuda"])
        self.update = True


    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {}
        params["model_name"] = str(self.model_name)
        params["prompt"] = str(self.prompt)
        params["conf_thres"] = str(self.conf_thres)
        params["conf_thres_text"] = str(self.conf_thres_text)
        params["cuda"] = str(self.cuda)
        return params


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferGroundingDino(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Create parameters class
        if param is None:
            self.set_param_object(InferGroundingDinoParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.model_weight = None
        self.model = None
        self.model_file_name = "groundingdino_swint_ogc.pth"
        self.config_file_name = "GroundingDINO_SwinT_OGC.py"
        self.url_base = "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        self.url_ext = "v0.1.0-alpha/groundingdino_swint_ogc.pth"
        

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1
    
    def transform_image(self, image):
        transform = T.Compose(
                [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        img_pil = Image.fromarray(image).convert("RGB")
        image_transformed, _ = transform(img_pil, None)
        return image_transformed

    def resize_bbox(self, image_source, boxes):
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        return xyxy

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get input :
        task_input = self.get_input(0)
        src_image = task_input.get_image()

        # Get parameters :
        param = self.get_param_object()

        if param.update or self.model is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")

            if param.model_name == "Swin-B":
                self.model_file_name = "groundingdino_swinb_cogcoor.pth"
                self.config_file_name = "GroundingDINO_SwinB_cfg.py"
                self.url_ext = "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"

            model_config = os.path.join(
                                os.path.dirname(
                                        os.path.realpath(__file__)),
                                        "GroundingDINO",
                                        "groundingdino",
                                        "config", 
                                        self.config_file_name
                                    )
            
            model_weigth = os.path.join(
                                os.path.dirname(
                                        os.path.realpath(__file__)),
                                        "weights",
                                        self.model_file_name,
                                    )
            
            # Download model weight if not exist
            weights_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
            if not os.path.isdir(weights_folder):
                os.mkdir(weights_folder)

            if not os.path.isfile(model_weigth):
                url = os.path.join(self.url_base, self.url_ext)
                print("Downloading model weight from {}".format(url))
                file_path = os.path.join(weights_folder, self.model_file_name)
                urllib.request.urlretrieve(url, file_path)
                print("Download completed!")
    
            self.model = load_model(
                                model_config_path=model_config,
                                model_checkpoint_path=model_weigth, 
                                device=self.device)

        image = self.transform_image(src_image)

        boxes, scores, phrases = predict(
                                    model=self.model, 
                                    image=image, 
                                    caption=param.prompt, 
                                    box_threshold=param.conf_thres, 
                                    text_threshold=param.conf_thres_text
                                )

        boxes_xyxy = self.resize_bbox(src_image, boxes)
 
        self.set_names(phrases)

        scores = scores.detach().cpu().numpy()
        index = 0
        for box, score in zip(boxes_xyxy, scores):
            cls = int(index)
            conf = score
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            x = float(box[0])
            y = float(box[1])
            self.add_object(index, cls, float(conf), x, y, w, h)
            index += 1

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferGroundingDinoFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_grounding_dino"
        self.info.short_description = "Inference of the Grounding DINO model"
        self.info.description = "The Algorithm proposes a zero-shot object grounding model "\
                                "that can localize objects in an image with a natural language query. " \
                                "Two models are available Swin-T (tiny) and Swin-B (Base). "\
                                "They have been trained on the COCO dataset. " \
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, " \
                            "Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, "\
                            "Jianwei and Su, Hang and Zhu, Jun and others"
        self.info.article = "Grounding dino: Marrying dino with grounded pre-training "\
                            "for open-set object detection"
        self.info.journal = "arXiv preprint arXiv:2303.05499"
        self.info.year = 2023
        self.info.license = "Apache License"
        # URL of documentation
        self.info.documentation_link = "https://github.com/IDEA-Research/GroundingDINO"
        # Code source repository
        self.info.repository = "https://github.com/IDEA-Research/GroundingDINO"
        # Keywords used for search
        self.info.keywords = "Object,Detection,Grounding,DINO,Zero Shot, Bert, Swin Transformer"

    def create(self, param=None):
        # Create process object
        return InferGroundingDino(self.info.name, param)
