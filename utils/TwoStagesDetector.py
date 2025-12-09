import os
import glob

import torch
from PIL import Image
import cv2
from graphic.RawDataProcessor import waterfall_spectrogram_optimized
from logger import colorful_logger
import json
import numpy as np
from benchmark import Classify_Model, Detection_Model, is_valid_file, raw_data_ext, image_ext
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock, Event
from collections import OrderedDict


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

        if self.S1.S1model:
            if save:
                with torch.no_grad:
                    res = self.S1.S1model.inference(source=source, save_dir=self.target_dir)
            else:
                source.seek(0)
                temp = np.asarray(bytearray(source.read()), dtype=np.uint8)
                temp = cv2.imdecode(temp, cv2.IMREAD_COLOR)
                res = self.S1.S1model.inference(source=temp)
            if not self.S2model:
                if save:
                    cv2.imwrite(self.save_path, res)
                else:
                    return res

        if self.S2model:
            if save: name = os.path.basename(source)[:-4]
            origin_image = Image.open(source).convert('RGB')
            preprocessed_image = self.S2model.preprocess(source)

            probability, predicted_class_name = self.S2model.forward(preprocessed_image)

            if not self.S1.S1model:
                res = self.S2model.add_result(res=predicted_class_name,
                                              probability=predicted_class_name,
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
        优化的原始数据处理流程：流水线处理、并行、不阻塞。
        
        处理流程：
        1. 输入整段原始信号
        2. 根据视频帧切分数据生成目标帧
        3. 对每一帧做FFT（数据复用）
        4. 并行生成时频图
        5. 一旦有时频图生成，立即用二阶段模型处理
        6. 按时间顺序重组为视频文件

        Parameters:
        - source (str): 原始数据路径
        - fft_size (int): FFT窗口大小
        - fs (float): 采样率
        - time_scale (int): 时间尺度
        - target_frame_gap (int): 目标帧间隔
        - num_workers (int): 并行工作线程数
        - fps (int): 输出视频帧率
        """
        name = os.path.splitext(os.path.basename(source))
        
        # 使用优化的瀑布图生成函数，返回带时间顺序的时频图
        self.logger.log_with_color(f"开始处理原始数据: {name[0]}")
        self.logger.log_with_color(f"正在生成时频图（并行处理，数据复用）...")
        
        # 生成时频图（并行、数据复用）
        images_list = waterfall_spectrogram_optimized(
            source, fft_size=fft_size, fs=fs, location='buffer', 
            time_scale=time_scale, target_frame_gap=target_frame_gap, 
            num_workers=num_workers
        )
        
        if not images_list:
            self.logger.log_with_color(f"警告: 未生成任何时频图")
            return
        
        self.logger.log_with_color(f"生成了 {len(images_list)} 个时频图，开始模型推理...")
        
        # 处理结果缓存（带时间顺序的哈希表）
        processed_results = OrderedDict()
        results_lock = Lock()
        video_initialized = False
        video = None
        
        def process_single_image(frame_idx, image_buffer):
            """处理单个时频图"""
            try:
                result = self.ImgProcessor(image_buffer, save=False)
                return frame_idx, result
            except Exception as e:
                self.logger.log_with_color(f"处理帧 {frame_idx} 时出错: {e}")
                return frame_idx, None
        
        # 并行处理所有时频图（不阻塞）
        with torch.no_grad():
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务（images_list已经按时间顺序排列）
                futures = {executor.submit(process_single_image, idx, img): idx 
                          for idx, img in enumerate(images_list)}
                
                # 按完成顺序收集结果（但保持时间顺序）
                completed_count = 0
                next_expected_frame = 0  # 下一个期望的帧索引
                
                for future in as_completed(futures):
                    frame_idx, result = future.result()
                    
                    if result is not None:
                        with results_lock:
                            processed_results[frame_idx] = result
                            completed_count += 1
                            
                            # 初始化视频写入器（在第一次获得结果时）
                            if not video_initialized:
                                height, width, layers = result.shape
                                video_name = name[0] + '_output.avi'
                                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                video_path = os.path.join(self.save_path, video_name)
                                video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                                video_initialized = True
                                self.logger.log_with_color(f"视频写入器已初始化: {width}x{height}, {fps}fps")
                            
                            # 一旦有新结果，立即写入视频（按时间顺序）
                            if video_initialized:
                                # 检查是否可以写入连续帧（从next_expected_frame开始）
                                while next_expected_frame in processed_results:
                                    video.write(processed_results[next_expected_frame])
                                    del processed_results[next_expected_frame]
                                    next_expected_frame += 1
                            
                            if completed_count % 10 == 0:
                                self.logger.log_with_color(f"已处理 {completed_count}/{len(images_list)} 帧，已写入 {next_expected_frame} 帧到视频")
        
        # 写入剩余的结果（按时间顺序）
        if video_initialized:
            for frame_idx in sorted(processed_results.keys()):
                if frame_idx >= next_expected_frame:
                    video.write(processed_results[frame_idx])
                    next_expected_frame = frame_idx + 1
        
        # 释放视频写入器
        if video_initialized and video is not None:
            video.release()
            self.logger.log_with_color(f"视频已保存: {os.path.join(self.save_path, name[0] + '_output.avi')}")
        
        self.logger.log_with_color(f"完成处理 {name[0]}，共处理 {len(images_list)} 帧")

    def DroneDetector(self, cfg):

        """Initializes the first stage model.

        Args:
            cfg: Configuration for the detector model.
        """

        self.S1 = Detection_Model(cfg)

    def DroneClassifier(self, cfg, weight_path, save=True):

        """Initializes the second stage model.

        Args:
            cfg: Configuration for the classifier model.
            weight_path: Path to the weights for the classifier model.
            save (bool): Whether to save the model.
        """

        self.S2model = Classify_Model(cfg=cfg, weight_path=weight_path)

    @property
    def set_logger(self):

        """
        Sets up the logger.

        Returns:
        - logger (colorful_logger): Logger instance.
        """

        logger = colorful_logger('Inference')
        return logger


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

    # 在图片上添加文字
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