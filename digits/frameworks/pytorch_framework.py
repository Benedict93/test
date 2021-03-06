# File: PyTorch_framework.py
from __future__ import absolute_import

import os
import re
import subprocess
import time
import tempfile

import flask

from .errors import NetworkVisualizationError
from .framework import Framework


import digits
from digits import utils
from digits.config import config_value
from digits.model.tasks import PyTorchTrainTask
from digits.utils import subclass, override


@subclass
class PyTorchFramework(Framework):

    """
    Defines required methods to interact with the PyTorch framework
    """

    # short descriptive name
    NAME = 'PyTorch'

    # identifier of framework class
    CLASS = 'PyTorch'

    # whether this framework can shuffle data during training
    CAN_SHUFFLE_DATA = True
    SUPPORTS_PYTHON_LAYERS_FILE = False
    SUPPORTS_TIMELINE_TRACING = False
    # under torch.optim package
    SUPPORTED_SOLVER_TYPES = ['SGD','NESTEROV', 'ADAGRAD','RMSPROP','ADADELTA', 'ADAM', 'SPARSEADAM','ADAMAX','ASGD','LBFGS', 'RPROP',]

    SUPPORTED_DATA_TRANSFORMATION_TYPES = ['MEAN_SUBTRACTION', 'CROPPING']
    SUPPORTED_DATA_AUGMENTATION_TYPES = ['FLIPPING', 'QUAD_ROTATION', 'ARBITRARY_ROTATION',
                                         'SCALING', 'NOISE', 'HSV_SHIFTING']

    def __init__(self):
        super(PyTorchFramework, self).__init__()
        # id must be unique
        self.framework_id = self.CLASS

    @override
    def create_train_task(self, **kwargs):
        """
        create train task
        """
        return PyTorchTrainTask(framework_id=self.framework_id, **kwargs)

    @override
    def get_standard_network_desc(self, network):
        """
        return description of standard network
        """
        networks_dir = os.path.join(os.path.dirname(digits.__file__), 'standard-networks', self.CLASS)

        for filename in os.listdir(networks_dir):
            path = os.path.join(networks_dir, filename)
            if os.path.isfile(path):
                match = None
                match = re.match(r'%s.py$' % network, filename)
                if match:
                    with open(path) as infile:
                        return infile.read()
        # return None if not found
        return None

    @override
    def get_network_from_desc(self, network_desc):
        """
        return network object from a string representation
        """
        # return the same string
        return network_desc

    @override
    def get_network_from_previous(self, previous_network, use_same_dataset):
        """
        return new instance of network from previous network
        """
        # note: use_same_dataset is ignored here because for Torch, DIGITS
        # does not change the number of outputs of the last linear layer
        # to match the number of classes in the case of a classification
        # network. In order to write a flexible network description that
        # accounts for the number of classes, the `nClasses` external
        # parameter must be used, see documentation.

        # return the same network description
        return previous_network

    @override
    def get_network_from_path(self, path):
        """
        return network object from a file path
        """
        with open(path, 'r') as f:
            network = f.read()
        return network

    @override
    def validate_network(self, data):
        """
        validate a network
        """
        return True

    @override
    def get_network_visualization(self, **kwargs):
        """
        return visualization of network
        """
        desc = kwargs['desc']
        # save network description to temporary file
        temp_network_handle, temp_network_path = tempfile.mkstemp(suffix='.py')
        os.write(temp_network_handle, desc)
        os.close(temp_network_handle)

        
