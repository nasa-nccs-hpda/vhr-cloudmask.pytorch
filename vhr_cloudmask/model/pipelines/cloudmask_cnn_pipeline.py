import os
import re
import sys
import time
import torch
import logging
import rasterio
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio.features as riofeat

from glob import glob

from pathlib import Path
from scipy import ndimage
from skimage import color
from functools import partial
from datetime import datetime
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from huggingface_hub import hf_hub_download

from vhr_cloudmask.model.config.cloudmask_config \
    import CloudMaskConfig as Config

from omnicloudmask import predict_from_array, predict_from_load_func, load_multiband
import omnicloudmask



CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
__status__ = "Production"


# -----------------------------------------------------------------------------
# class CloudMaskPipeline
# -----------------------------------------------------------------------------
class CloudMaskPipeline(object):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(
                self,
                config_filename: str = None,
                model_filename: str = None,
                output_dir: str = None,
                inference_regex_list: str = None,
                band_order: str = None,
                postprocessing_steps: list = None,
                overwrite: bool = False,
                default_config: str = 'templates/cloudmask_default.yaml',
                logger=None
            ):
        """Constructor method
        """
        logging.info('Initializing CloudMaskPipeline')

        # Configuration file intialization
        if config_filename is None:
            config_filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                default_config)
            logging.info(f'Loading default config: {config_filename}')

        # load configuration into object
        self.conf = self._read_config(config_filename, Config)

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        # overwrite option
        self.conf.overwrite = overwrite

        # rewrite model filename option if given from CLI
        if model_filename is not None:
            assert os.path.exists(model_filename), \
                f'{model_filename} does not exist.'
            self.conf.model_filename = model_filename

        # rewrite output directory if given from CLI
        if output_dir is not None:
            self.conf.inference_save_dir = output_dir

        # rewrite inference regex list
        if inference_regex_list is not None:
            self.conf.inference_regex_list = inference_regex_list

        # rewrite input bands if given from CLI
        if band_order is not None:
            self.conf.band_order = band_order

        # rewrite postprocessing steps if given from CLI
        if postprocessing_steps is not None:
            self.conf.postprocessing_steps = postprocessing_steps

        # Create output directories
        os.makedirs(self.conf.inference_save_dir, exist_ok=True)
        logging.info(f'Output dir: {self.conf.inference_save_dir}')

        # save configuration into the model directory
        logging.info(
            f'Output GeoTIFF driver: {self.conf.prediction_driver}')

    # -------------------------------------------------------------------------
    # _read_config
    # -------------------------------------------------------------------------
    def _read_config(self, filename: str, config_class=Config):
        """
        Read configuration filename and initiate objects
        """
        # Configuration file initialization
        schema = OmegaConf.structured(config_class)
        conf = OmegaConf.load(filename)
        try:
            conf = OmegaConf.merge(schema, conf)
        except BaseException as err:
            sys.exit(f"ERROR: {err}")
        return conf

    # -------------------------------------------------------------------------
    # _set_logger
    # -------------------------------------------------------------------------
    def _set_logger(self):
        """
        Set logger configuration.
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # set console output
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # set filename output
        log_filename = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'

        if self.conf.inference_save_dir is not None:
            os.makedirs(self.conf.inference_save_dir, exist_ok=True)
            log_filename = os.path.join(
                self.conf.inference_save_dir, log_filename)
        else:
            os.makedirs('vhr-cloudmask-logs', exist_ok=True)
            log_filename = os.path.join(
                'vhr-cloudmask-logs', log_filename)

        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # remove the root logger
        logger.handlers.pop(0)
        return logger

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def get_filenames(self, data_regex: str) -> list:
        """
        Get filename from list of regexes
        """
        # get the paths/filenames of the regex
        filenames = []
        if isinstance(data_regex, list) or isinstance(data_regex, ListConfig):
            for regex in data_regex:
                filenames.extend(glob(regex))
        else:
            filenames = glob(data_regex)
        assert len(filenames) > 0, \
            f'No files under {data_regex}, provide inference_regex_list.'
        return sorted(filenames)

    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    def predict(self) -> None:
        """This will perform inference on a list of GeoTIFF files provided
        as a list of regexes from the CLI.

        :return: None, outputs GeoTIFF cloudmask files to disk.
        :rtype: None
        """

        logging.info('Starting prediction stage')

        # Gather filenames to predict
        if len(self.conf.inference_regex_list) > 0:
            data_filenames = self.get_filenames(self.conf.inference_regex_list)
        else:
            data_filenames = self.get_filenames(self.conf.inference_regex)
        logging.info(f'{len(data_filenames)} files to predict')

        # Define prediction function
        load_maxar_pgc_4band_10m = partial(
            load_multiband, resample_res=10.0, band_order=[3, 2, 4])

        # iterate files, create lock file to avoid predicting the same file
        for filename in sorted(data_filenames):

            # start timer
            start_time = time.time()

            # set prediction output lock filename
            output_stem = os.path.join(
                self.conf.inference_save_dir,
                Path(filename).stem
            )

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_stem}.lock'

            # predict only if file does not exist and no lock file
            if self.conf.overwrite or (
                        not os.path.isfile(lock_filename) and
                        len(glob(f"{output_stem}*.tif")) == 0
                    ):

                try:

                    logging.info(f'Starting to predict {filename}')

                    # create lock file
                    open(lock_filename, 'w').close()

                    # open filename
                    image = rxr.open_rasterio(filename)
                    logging.info(f'Prediction shape: {image.shape}')

                    # check bands in imagery, do not proceed if one band
                    if image.shape[0] == 1:
                        logging.info(
                            'Skipping file because of non sufficient bands')
                        continue

                except rasterio.errors.RasterioIOError:
                    logging.info(f'Skipped {filename}, probably corrupted.')
                    continue

                output_filename = predict_from_load_func(
                    scene_paths=[filename],
                    load_func=load_maxar_pgc_4band_10m,
                    inference_dtype=self.conf.inference_dtype,
                    no_data_value=self.conf.no_data_value,
                    output_dir=self.conf.inference_save_dir,
                    inference_device=self.conf.inference_device,
                    mosaic_device=self.conf.mosaic_device,
                    overwrite=self.conf.overwrite,
                    batch_size=self.conf.batch_size,
                )

                with torch.no_grad():
                    torch.cuda.empty_cache()

                # rename output filename
                output_filename_path = Path(output_filename[0])
                new_name = re.sub(
                    r"[_-]?OCM.*$", "_cloudmask.tif",
                    output_filename_path.name
                )

                output_filename_path.rename(
                    output_filename_path.with_name(new_name))

                # apply default postprocessing
                # prediction = self.postprocessing(
                #    prediction, self.conf.postprocessing_steps)

                # get cloud metrics for metadata
                # cloud_metadata = self.calc_metadata(prediction)

                # Drop image band to allow for a merge of mask
                # image = image.drop(
                #    dim="band",
                #    labels=image.coords["band"].values[1:],
                # )

                # Get metadata to save raster prediction
                # prediction = xr.DataArray(
                #    np.expand_dims(prediction, axis=-1),
                #    name=self.conf.experiment_type,
                #    coords=image.coords,
                #    dims=image.dims,
                #    attrs=image.attrs
                # )

                # Add metadata to raster attributes
                # prediction.attrs['long_name'] = (self.conf.experiment_type)
                # prediction.attrs['model_name'] = (model_filename)

                # add cloud metadata to raster attributes
                # for mkey, mvalue in cloud_metadata.items():
                #    prediction.attrs[mkey] = (mvalue)

                # transpose prediction for saving output
                # prediction = prediction.transpose("band", "y", "x")

                # Set nodata values on mask
                # nodata = prediction.rio.nodata
                # prediction = prediction.where(image != nodata)
                # prediction.rio.write_nodata(
                #    self.conf.prediction_nodata, encoded=True, inplace=True)

                # Save output raster file to disk
                # prediction.rio.to_raster(
                #    output_filename,
                #    BIGTIFF="IF_SAFER",
                #    compress=self.conf.prediction_compress,
                #    driver=self.conf.prediction_driver,
                #    dtype=self.conf.prediction_dtype
                # )
                # del prediction

                # delete lock file
                try:
                    os.remove(lock_filename)
                except FileNotFoundError:
                    logging.info(f'Lock file not found {lock_filename}')
                    continue

                logging.info(f'Finished processing {output_filename}')
                logging.info(f"{(time.time() - start_time)/60} min")

            # This is the case where the prediction was already saved
            else:
                output_filename = glob(f"{output_stem}*.tif")
                if len(output_filename) > 0:
                    logging.info(
                        f'{output_filename[0]} already exists, skipping.')
                else:
                    logging.info(
                        f'{lock_filename} file exists, delete and restart '
                        'prediction, or set overwrite=True.'
                    )

        return

    # -------------------------------------------------------------------------
    # postprocessing
    # -------------------------------------------------------------------------
    def postprocessing(
                self,
                prediction: np.ndarray,
                postprocessing_steps: list = []
            ) -> np.ndarray:

        # sieve clearing of small objects
        if 'sieve' in postprocessing_steps:
            riofeat.sieve(prediction, 800, prediction, None, 8)

        # cloud smoothing
        if 'smooth' in postprocessing_steps:
            prediction = ndimage.median_filter(prediction, size=20)

        # binary fill holes
        if 'fill' in postprocessing_steps:
            prediction = ndimage.binary_fill_holes(prediction).astype(int)

        # dilate
        if 'dilate' in postprocessing_steps:

            # get prediction and set cv2 format
            prediction = np.uint8(np.squeeze(prediction) * 255)
            _, thresh = cv2.threshold(prediction, 127, 255, 0)

            # gather contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # set output contours
            conts = np.zeros((thresh.shape[0], thresh.shape[1], 3))

            # iterate over each contour and increase size
            for c in contours:
                c = self.scale_contour(c, 1.12)  # 1.05 looks decent
                conts = cv2.drawContours(
                    conts, [c], -1, (0, 255, 0), cv2.FILLED)

            # converting mask into raster
            prediction = color.rgb2gray(conts).astype('int16')
            prediction[prediction > 0] = 1

        return prediction

    # -------------------------------------------------------------------------
    # calc_metadata
    # -------------------------------------------------------------------------
    def calc_metadata(self, prediction: np.ndarray) -> dict:

        # set metadata directory for cloud metadata
        metadata_dict = {}

        #  cloud percentage
        metadata_dict['pct_cloudcover_total'] = self.calc_cloud_percentage(
            prediction)

        return metadata_dict

    # -------------------------------------------------------------------------
    # calc_metadata
    # -------------------------------------------------------------------------
    def calc_cloud_percentage(self, prediction: np.ndarray) -> float:

        # get unique values per class
        unique, counts = np.unique(prediction, return_counts=True)
        unique_dict = dict(zip(unique, counts))

        # get cloudy pixels
        try:
            non_cloud_pixels = unique_dict[0]
        except (IndexError, KeyError):
            non_cloud_pixels = 0

        # get cloudy pixels
        try:
            cloud_pixels = unique_dict[1]
        except (IndexError, KeyError):
            cloud_pixels = 0

        # percent cloud cover
        pct_cloud_cover = round(
            100 * (cloud_pixels / (non_cloud_pixels + cloud_pixels)), 2)

        return pct_cloud_cover

    # -------------------------------------------------------------------------
    # scale_contour
    # -------------------------------------------------------------------------
    def scale_contour(self, cnt: np.ndarray, scale: float) -> np.ndarray:
        M = cv2.moments(cnt)
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        except ZeroDivisionError:
            cx, cy = 0, 0

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        return cnt_scaled.astype(np.int32)
