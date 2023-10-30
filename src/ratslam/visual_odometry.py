from typing import List, Tuple

import numpy as np
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

# note: you can't use "from .util import *" because lava will have an aneurysm
from ratslam.util import compare_segments


class VisualOdometry(AbstractProcess):
    def __init__(self, image_shape: Tuple[int, int]) -> None:
        super().__init__()
        self.img_in = InPort(shape=image_shape)

        self.vtrans_vrot_out = OutPort(shape=(2,))


@implements(proc=VisualOdometry, protocol=LoihiProtocol)
@requires(CPU)
class PyVisualOdometryModel(PyLoihiProcessModel):
    img_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    prev_img_1d = None

    vtrans_vrot_out: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, float, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

    def post_guard(self):
        return self.img_in.probe()

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above
        returns True.
        """

        VISUAL_ODO_SHIFT_MATCH = 80
        VTRANS_SCALE = 10
        VROT_SCALE = 1  # (CAMERA_FOV_DEG/img.shape[1])*np.pi/180.0

        img = self.img_in.recv()
        img_1d = np.sum(img, axis=0)
        img_1d = img_1d / np.sum(img_1d)

        if self.prev_img_1d is None:
            self.prev_img_1d = img_1d
            return

        offset, diff = compare_segments(
            img_1d,
            self.prev_img_1d,
            VISUAL_ODO_SHIFT_MATCH
        )
        vtrans = diff*VTRANS_SCALE
        vrot = offset*VROT_SCALE

        arr_out = np.array([vtrans, vrot])
        self.vtrans_vrot_out.send(arr_out)

        self.prev_img_1d = img_1d

    def run_spk(self):
        # print("visual_odometry spike")
        pass
