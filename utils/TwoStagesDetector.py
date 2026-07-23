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

        """Run the two-stage pipeline on a single spectrogram image path.

        Stage 1 (any detector backend) draws boxes; stage 2 classifies the whole
        image and overlays the label. Works for detector-only, classifier-only,
        or both. Returns the annotated image when `save` is False, else writes it
        under `save_path`.
        """

        name = os.path.splitext(os.path.basename(source))[0]
        out_path = os.path.join(self.save_path, name + '.jpg')

        # Stage 1: detection (backend-agnostic) -> annotated BGR image
        det_img = self.S1.detect(source) if self.S1 is not None else None

        # Stage 2: whole-image classification
        if self.S2model is not None:
            preprocessed_image = self.S2model.preprocess(source)
            probability, predicted_class_name = self.S2model.forward(preprocessed_image)

            if det_img is None:
                # classification only -> annotate the original image (PIL)
                origin_image = Image.open(source).convert('RGB')
                res = self.S2model.add_result(res=predicted_class_name,
                                              probability=probability,
                                              image=origin_image)
                if save:
                    res.save(out_path)
                    return
                return res

            det_img = put_res_on_img(det_img, predicted_class_name, probability=probability)

        if det_img is None:
            return
        if save:
            cv2.imwrite(out_path, det_img)
            return
        return det_img

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
        name = os.path.splitext(os.path.basename(source))[0]

        self.logger.log_with_color(f"Starting to process raw data: {name}")
        self.logger.log_with_color("Generating spectrograms (parallel processing, data reuse)...")

        # Generate spectrograms (parallel, data reuse) -> time-ordered buffers
        images_list = waterfall_spectrogram_optimized(
            source, fft_size=fft_size, fs=fs, location='buffer',
            time_scale=time_scale, target_frame_gap=target_frame_gap,
            num_workers=num_workers
        )

        if not images_list:
            self.logger.log_with_color("Warning: No spectrograms generated")
            return

        n = len(images_list)
        self.logger.log_with_color(f"Generated {n} spectrograms, starting model inference...")

        # Decode each frame buffer per library (was decoded up to 3x per frame):
        # a BGR array for stage-1 detection + video, and — only when stage 2 is
        # present — an RGB PIL image via the same PIL path the classifier used.
        need_pil = self.S2model is not None
        bgr_frames, pil_frames = [], []
        for buf in images_list:
            buf.seek(0)
            bgr_frames.append(cv2.imdecode(np.frombuffer(buf.read(), dtype=np.uint8), cv2.IMREAD_COLOR))
            if need_pil:
                buf.seek(0)
                pil_frames.append(Image.open(buf).convert('RGB'))

        # --- Stage 2: batch-classify every frame (whole-frame classification,
        #     independent of stage-1 detections) instead of one frame at a time.
        cls_results = [None] * n
        if self.S2model is not None:
            s2 = self.S2model
            bs = int(s2.cfg.get('batch_size') or 8)
            with torch.no_grad():
                for i in range(0, n, bs):
                    chunk = pil_frames[i:i + bs]
                    batch = torch.stack([s2._transform(im) for im in chunk]).to(s2.device, non_blocking=True)
                    with s2._autocast():
                        outputs = s2.model(batch)
                    probs = torch.softmax(outputs.float(), dim=1)
                    conf, idx = probs.max(dim=1)
                    for k in range(len(chunk)):
                        cls_results[i + k] = (s2._idx_to_name[int(idx[k].item())], float(conf[k].item()) * 100)

        # --- Stage 1: batch-detect frames, then compose result + write video in
        #     time order (no thread pool: GPU forwards don't parallelise across
        #     Python threads, and batching is what actually speeds this up).
        has_s1 = self.S1 is not None
        video = None
        video_path = os.path.join(self.save_path, name + '_output.avi')
        det_bs = 8
        with torch.no_grad():
            for i in range(0, n, det_bs):
                chunk_bgr = bgr_frames[i:i + det_bs]
                det_imgs = self.S1.detect_batch(chunk_bgr) if has_s1 else chunk_bgr
                for j, det_img in enumerate(det_imgs):
                    cls = cls_results[i + j]
                    if cls is not None:
                        det_img = put_res_on_img(det_img, cls[0], probability=cls[1])
                    if video is None:
                        height, width = det_img.shape[:2]
                        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
                        self.logger.log_with_color(f"Video writer initialized: {width}x{height}, {fps}fps")
                    video.write(det_img)
                self.logger.log_with_color(f"Processed {min(i + det_bs, n)}/{n} frames")

        if video is not None:
            video.release()
            self.logger.log_with_color(f"Video saved: {video_path}")

        self.logger.log_with_color(f"Completed processing {name}, processed {n} frames")

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