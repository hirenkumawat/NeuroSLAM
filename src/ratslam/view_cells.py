
from dataclasses import dataclass
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

from ratslam.util import compare_segments


@dataclass
class ViewCell:
    img_1d: np.ndarray
    x_pc: float
    y_pc: float
    th_pc: float
    decay: float


class ViewCells(AbstractProcess):
    def __init__(self, image_shape: Tuple[int, int]) -> None:
        super().__init__()
        self.img_in = InPort(shape=image_shape)
        self.pose_in = InPort(shape=(3,))

        self.cell_out = OutPort(shape=(4,))


@implements(proc=ViewCells, protocol=LoihiProtocol)
@requires(CPU)
class PyViewCellsModel(PyLoihiProcessModel):
    img_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    pose_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)

    cell_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)

    cells = []

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

    def post_guard(self):
        return self.img_in.probe() and self.pose_in.probe()

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above
        returns True.
        """
        VT_MATCH_THRESHOLD = .3 # 0.054
        VT_ACTIVE_DECAY = 1.0

        img = self.img_in.recv()
        img_1d = create_template(img)
        pose = self.pose_in.recv()

        cell_similarities = np.zeros(len(self.cells))
        for idx, cell in enumerate(self.cells):
            cell_similarities[idx] = get_similarity(img_1d, cell.img_1d)
        # if len(self.cells) != 0:
        #     print(np.min(cell_similarities)*img_1d.size)

        if len(self.cells) == 0 or np.min(cell_similarities)*img_1d.size > VT_MATCH_THRESHOLD:
            new_cell = ViewCell(
                img_1d,
                x_pc=pose[0],
                y_pc=pose[1],
                th_pc=pose[2],
                decay=VT_ACTIVE_DECAY
            )
            self.cells.append(new_cell)
            self.prev_cell = new_cell
            print("new cell")
            return new_cell

        i = np.argmin(cell_similarities)
        cell = self.cells[i]
        cell.decay += VT_ACTIVE_DECAY

        print("old cell")
        self.prev_cell = cell
        return cell


def create_template(img: np.ndarray) -> np.ndarray:
    img_1d = np.sum(img, axis=0)
    return img_1d / np.sum(img_1d)


def get_similarity(seg1: np.ndarray, seg2: np.ndarray) -> float:
    VT_SHIFT_MATCH = 25
    offset, dist = compare_segments(seg1, seg2, VT_SHIFT_MATCH)
    return dist
