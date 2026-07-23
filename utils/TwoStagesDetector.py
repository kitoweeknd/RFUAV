from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image
import torch

from graphic.RawDataProcessor import waterfall_spectrogram_optimized
from .benchmark import (
    ClassifyModel,
    DetectionModel,
    image_ext,
    is_valid_file,
    raw_data_ext,
)
from .logger import colorful_logger


class TwoStagesDetector:

    def __init__(self, cfg: str = ''):

        """A data flow processing class for a two-stage model, providing public interfaces.

        Args:
            cfg (str): Path to the configuration file.
        """

        self.logger = colorful_logger('Inference')
        det, cla, save_path, target_dir = load_model_from_json(cfg)
        self.det = det
        self.cla = cla
        self.save_path = save_path
        self.target_dir = target_dir

        # Stage models: only the stage(s) selected below get instantiated,
        # so default both to None and guard reads for single-stage configs.
        self.S1 = None
        self.S2model = None

        if not cla and det:
            self.DroneDetector(cfg=det)
        elif not det and cla:
            self.DroneClassifier(cfg=cla['cfg'], weight_path=cla['weight_path'], save=True)
        elif det and cla:
            self.DroneDetector(cfg=det)
            self.DroneClassifier(cfg=cla['cfg'], weight_path=cla['weight_path'], save=True)
        else:
            raise ValueError("No model is selected")

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.logger.log_with_color(f"Saving results to: {save_path}")

        if not os.path.exists(target_dir):
            raise ValueError(f"Source {target_dir} dose not exit")

        # dir detect
        if os.path.isdir(target_dir):
            data_list = glob.glob(os.path.join(target_dir, '*'))

            for data in data_list:
                # detect images in dir
                if is_valid_file(data, image_ext):
                    self.ImgProcessor(data)
                # detect raw datas in dir
                elif is_valid_file(data, raw_data_ext):
                    self.RawdataProcess(data)
                else:
                    continue

        # detect single image
        elif is_valid_file(target_dir, image_ext):
            self.ImgProcessor(target_dir)

        # detect single pack of raw data
        elif is_valid_file(target_dir, raw_data_ext):
            self.RawdataProcess(target_dir)

    def ImgProcessor(self, source, save=True):

        """Processes an image source using the first and second stage models.

        Args:
            source: The image source to be processed.
            save (bool): Whether to save the processed image.

        Returns:
            Processed image if `save` is False, otherwise None.
        """

        if self.S1 is not None and self.S1.S1model:
            if save:
                with torch.no_grad():
                    res = self.S1.S1model.inference(source=source, save_dir=self.target_dir)
            else:
                source.seek(0)
                temp = np.asarray(bytearray(source.read()), dtype=np.uint8)
                temp = cv2.imdecode(temp, cv2.IMREAD_COLOR)
                res = self.S1.S1model.inference(source=temp)
            if self.S2model is None:
                if save:
                    cv2.imwrite(self.save_path, res)
                else:
                    return res

        if self.S2model is not None:
            if save: name = os.path.basename(source)[:-4]
            origin_image = Image.open(source).convert('RGB')
            preprocessed_image = self.S2model.preprocess(source)

            probability, predicted_class_name = self.S2model.forward(preprocessed_image)

            if self.S1 is None or not self.S1.S1model:
                res = self.S2model.add_result(res=predicted_class_name,
                                              probability=probability,
                                              image=origin_image)
                if save:
                    res.save(os.path.join(self.save_path, name + '.jpg'))
                else:
                    return res

            else:
                res = put_res_on_img(res, predicted_class_name, probability=probability)

                if save:
                    cv2.imwrite(self.save_path, res)
                else:
                    return res

    def RawdataProcess(self, source, fft_size=256, fs=100e6, time_scale=39062, 
                       target_frame_gap=150, num_workers=4, fps=30):
        """
        Optimized raw data processing pipeline: pipelined processing, parallel, non-blocking.
        
        Processing flow:
        1. Input entire raw signal segment
        2. Generate target frames based on video frame segmentation
        3. Perform FFT on each frame (data reuse)
        4. Generate spectrograms in parallel
        5. Once spectrogram is generated, immediately process with two-stage model
        6. Reassemble into video file in chronological order

        Parameters:
        - source (str): Raw data path
        - fft_size (int): FFT window size
        - fs (float): Sampling rate
        - time_scale (int): Time scale
        - target_frame_gap (int): Target frame gap
        - num_workers (int): Number of parallel worker threads
        - fps (int): Output video frame rate
        """
        name = os.path.splitext(os.path.basename(source))
        
        # Use optimized waterfall spectrogram generation function, returns time-ordered spectrograms
        self.logger.log_with_color(f"Starting to process raw data: {name[0]}")
        self.logger.log_with_color(f"Generating spectrograms (parallel processing, data reuse)...")
        
        # Generate spectrograms (parallel, data reuse)
        images_list = waterfall_spectrogram_optimized(
            source, fft_size=fft_size, fs=fs, location='buffer', 
            time_scale=time_scale, target_frame_gap=target_frame_gap, 
            num_workers=num_workers
        )
        
        if not images_list:
            self.logger.log_with_color(f"Warning: No spectrograms generated")
            return
        
        self.logger.log_with_color(f"Generated {len(images_list)} spectrograms, starting model inference...")
        
        # Result cache (hash table maintaining time order)
        processed_results = OrderedDict()
        results_lock = Lock()
        video_initialized = False
        video = None
        
        def process_single_image(frame_idx, image_buffer):
            """Process single spectrogram"""
            try:
                result = self.ImgProcessor(image_buffer, save=False)
                return frame_idx, result
            except Exception as e:
                self.logger.log_with_color(f"Error processing frame {frame_idx}: {e}")
                return frame_idx, None
        
        # Process all spectrograms in parallel (non-blocking)
        with torch.no_grad():
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks (images_list is already sorted by time order)
                futures = {executor.submit(process_single_image, idx, img): idx 
                          for idx, img in enumerate(images_list)}
                
                # Collect results in completion order (but maintain time order)
                completed_count = 0
                next_expected_frame = 0  # Next expected frame index
                
                for future in as_completed(futures):
                    frame_idx, result = future.result()
                    
                    if result is not None:
                        with results_lock:
                            processed_results[frame_idx] = result
                            completed_count += 1
                            
                            # Initialize video writer (when first result is obtained)
                            if not video_initialized:
                                height, width, layers = result.shape
                                video_name = name[0] + '_output.avi'
                                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                video_path = os.path.join(self.save_path, video_name)
                                video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                                video_initialized = True
                                self.logger.log_with_color(f"Video writer initialized: {width}x{height}, {fps}fps")
                            
                            # Once new result arrives, write to video immediately (in time order)
                            if video_initialized:
                                # Check if consecutive frames can be written (starting from next_expected_frame)
                                while next_expected_frame in processed_results:
                                    video.write(processed_results[next_expected_frame])
                                    del processed_results[next_expected_frame]
                                    next_expected_frame += 1
                            
                            if completed_count % 10 == 0:
                                self.logger.log_with_color(f"Processed {completed_count}/{len(images_list)} frames, written {next_expected_frame} frames to video")
        
        # Write remaining results (in time order)
        if video_initialized:
            for frame_idx in sorted(processed_results.keys()):
                if frame_idx >= next_expected_frame:
                    video.write(processed_results[frame_idx])
                    next_expected_frame = frame_idx + 1
        
        # Release video writer
        if video_initialized and video is not None:
            video.release()
            self.logger.log_with_color(f"Video saved: {os.path.join(self.save_path, name[0] + '_output.avi')}")
        
        self.logger.log_with_color(f"Completed processing {name[0]}, processed {len(images_list)} frames")

    def DroneDetector(self, cfg):

        """Initializes the first stage model.

        Args:
            cfg: Configuration for the detector model.
        """

        self.S1 = DetectionModel(cfg)

    def DroneClassifier(self, cfg, weight_path, save=True):

        """Initializes the second stage model.

        Args:
            cfg: Configuration for the classifier model.
            weight_path: Path to the weights for the classifier model.
            save (bool): Whether to save the model.
        """

        self.S2model = ClassifyModel(cfg=cfg, weight_path=weight_path)


def load_model_from_json(cfg):

    """Loads configuration from a JSON file.

    Args:
        cfg (str): Path to the configuration file.

    Returns:
        Tuple containing detector configuration, classifier configuration, save path, and target directory.
    """

    with open(cfg, 'r') as f:
        _ = json.load(f)
        return _['detector'] if 'detector' in _ else None, _['classifier'] if 'classifier' in _ else None, _['save_dir'], _['target_dir']


def put_res_on_img(img,
                   text,
                   probability=0.0,
                   position=(20, 60),
                   font_scale=1,
                   color=(0, 0, 0),
                   thickness=3):

    """Adds text result on an image.

    Args:
        img: Image to add text to.
        text (str): Text to add.
        probability (float): Probability value to display alongside the text.
        position (tuple): Position of the text on the image.
        font_scale (int): Font scale of the text.
        color (tuple): Color of the text.
        thickness (int): Thickness of the text.

    Returns:
        Image with added text.
    """

    # Add text on image
    cv2.putText(img=img,
                text=text + f" {probability:.2f}%",
                org=position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA)

    return img


# for test ------------------------------------------------------------------------------------------------------------
def main():
    cfg_path = '../example/two_stage/sample.json'
    TwoStagesDetector(cfg=cfg_path)


if __name__ == '__main__':
    main()