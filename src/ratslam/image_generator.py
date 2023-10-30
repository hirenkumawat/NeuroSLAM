from typing import List, Tuple

import cv2
import numpy as np
# Import decorators
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

# this is helpful:
# https://lava-nc.org/lava/notebooks/end_to_end/tutorial01_mnist_digit_classification.html


class ImageGenerator(AbstractProcess):
    def __init__(self, video_shape: tuple, num_steps_per_image: int = 128, **kwargs) -> None:
        """
        video_shape: (height, width)
        """
        super().__init__(**kwargs)

        self.num_steps_per_image = Var(shape=(1,), init=num_steps_per_image)
        self.cur_img = Var(shape=video_shape)
        self.img_out = OutPort(shape=video_shape)


@implements(proc=ImageGenerator, protocol=LoihiProtocol)
@requires(CPU)
class PyImageGeneratorModel(PyLoihiProcessModel):
    num_steps_per_image: int = LavaPyType(int, int, precision=32)
    cur_img: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    img_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        if proc_params.get('video_path') is None:
            raise Exception(
                "video path wasn't passed as kwarg to the abstract process")
        video_path = proc_params['video_path']
        self.video_data = cv2.VideoCapture(video_path)

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step % self.num_steps_per_image == 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above
        returns True.
        """
        _, frame = self.video_data.read()
        self.cur_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img_out.send(self.cur_img)

    # we will likely need to implement the following
    # when we change to using spiking

    def run_spk(self):
        # print("spikee")
        pass
