import logging
from abc import ABC, abstractmethod

import torch

from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor


logger = logging.getLogger(__name__)


class BasePyTorchModel(IFreqaiModel, ABC):
    """
    Base class for PyTorch type models.
    User *must* inherit from this class and set fit() and predict() and
    data_convertor property.
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs["config"])
        self.dd.model_type = "pytorch"
        self.device = self._select_device()
        test_size = self.freqai_info.get("data_split_parameters", {}).get("test_size")
        self.splits = ["train", "test"] if test_size != 0 else ["train"]
        self.window_size = self.freqai_info.get("conv_width", 1)

    @staticmethod
    def _select_device() -> str:
        """Choose an accelerator that the installed PyTorch build can execute on."""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"

        if not torch.cuda.is_available():
            return "cpu"

        try:
            capability = torch.cuda.get_device_capability(0)
            device_arch = f"sm_{capability[0]}{capability[1]}"
            supported_arches = set(torch.cuda.get_arch_list())
            if supported_arches and device_arch not in supported_arches:
                logger.warning(
                    "CUDA device architecture %s is not supported by this PyTorch build "
                    "(%s); falling back to CPU.",
                    device_arch,
                    ", ".join(sorted(supported_arches)),
                )
                return "cpu"

            # CUDA can be reported as available even when a kernel cannot run on it.
            torch.zeros(1, device="cuda")
        except (RuntimeError, torch.AcceleratorError) as exc:
            logger.warning("CUDA is unavailable for PyTorch (%s); falling back to CPU.", exc)
            return "cpu"

        return "cuda"

    @property
    @abstractmethod
    def data_convertor(self) -> PyTorchDataConvertor:
        """
        a class responsible for converting `*_features` & `*_labels` pandas dataframes
        to pytorch tensors.
        """
        raise NotImplementedError("Abstract property")
