from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class CloudMaskConfig:

    # test classes definition for validation
    test_classes: List[str] = field(
        default_factory=lambda: [
            'no-cloud', 'thick-cloud', 'thin-cloud', 'cloud-shadow']
    )

    # resolution expected by omni cloudmask
    resample_res: float = 10.0

    # band order expected from TOA imagery
    band_order: List[int] = field(
        default_factory=lambda: [3, 2, 4]
    )

    # dtype of inference output
    inference_dtype: str = "bf16"

    # nodata value of inference output
    no_data_value: int = -10001

    # inference device
    inference_device: str = 'cuda'

    # mosaic device
    mosaic_device: str = 'cpu'

    # overwrite output files
    overwrite: bool = False

    # prediction batch size
    batch_size: int = 1

    # rewrite default output to be COG
    prediction_driver: str = 'COG'

    # postprocessing settings
    postprocessing_steps: List[str] = field(
        default_factory=lambda: ['sieve', 'smooth', 'fill', 'dilate']
    )

    inference_save_dir: str = 'vhr-cloudmask-outputs'

    # List regex to find rasters to predict (multiple locations)
    inference_regex_list: Optional[List[str]] = field(
        default_factory=lambda: [])