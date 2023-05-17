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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_grounding_dino.infer_grounding_dino_process import InferGroundingDinoParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferGroundingDinoWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferGroundingDinoParam()
        else:
            self.parameters = param

        
        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Models name
        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model")
        self.combo_model.addItem("Swin-T")
        self.combo_model.addItem("Swin-B")
        self.combo_model.setCurrentText(self.parameters.model_name)

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)

        # Confidence thresholds
        self.spin_conf_thres_box = pyqtutils.append_double_spin(self.grid_layout, "Confidence threshold boxes",
                                                          self.parameters.conf_thres,
                                                          min=0., max=1., step=0.01, decimals=2)
        
        self.spin_conf_thres_text = pyqtutils.append_double_spin(self.grid_layout, "Confidence threshold text",
                                                    self.parameters.conf_thres_text,
                                                    min=0., max=1., step=0.01, decimals=2)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)
        
    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.conf_thres = self.spin_conf_thres_box.value()
        self.parameters.conf_thres_text = self.spin_conf_thres_text.value()
        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferGroundingDinoWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_grounding_dino"

    def create(self, param):
        # Create widget object
        return InferGroundingDinoWidget(param, None)
