# Tool to process raw data
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免多线程问题
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import numpy as np
import os
from io import BytesIO
import imageio
from PIL import Image
from typing import Union
from scipy.fft import fft
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import OrderedDict
import queue


class RawDataProcessor:
    """transform raw data into images, video, and save the result locally
    func:
    - TransRawDataintoSpectrogram() can process raw data in batches and save the results locally
    - TransRawDataintoVideo() can transform raw data into video and save the result locally
    - ShowSpectrogram() can show the spectrum of the raw data in prev 0.1s to check the raw data quicly
    """

    def TransRawDataintoSpectrogram(self,
                                    fig_save_path: str,
                                    data_path: str,
                                    sample_rate: Union[int, float] = 100e6,
                                    stft_point: int = 2048,
                                    duration_time: float = 0.1,
                                    file_type=np.float32
                                    ):
        """transform the raw data into spectromgrams and save the results locally
        :param fig_save_path: the target dir path to save the image result.
        :param data_path: the input raw data dir path.
        :param sample_rate: the simple rate when collect the raw frequency data using USRP, ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
        :param stft_point: the STFT points using in STFT transformation, you better set this to 2**n (n is a Non-negative integer).
        :param duration_time: the duration time of single spectromgram.
        """
        DrawandSave(fig_save_path=fig_save_path, file_path=data_path, fs=sample_rate,
                    stft_point=stft_point, duration_time=duration_time, file_type=file_type)

    def TransRawDataintoVideo(self,
                              save_path: str,
                              data_path: str,
                              sample_rate: Union[int, float] = 100e6,
                              stft_point: int = 2048,
                              duration_time: float = 0.1,
                              fps: int = 5,
                              file_type=np.float32
                              ):
        """transform the raw data into video and save the result locally
        :param save_path: the target dir path to save the image result.
        :param data_path: the input raw data dir path.
        :param sample_rate: the simple rate when collect the raw frequency data using USRP, ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
        :param stft_point: the STFT points using in STFT transformation, you better set this to 2**n (n is a Non-negative integer). ref: https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT
        :param duration_time: the duration time of single spectromgram.
        :param fps: control the fps of generated video.
        """
        save_as_video(datapack=data_path, save_path=save_path, fs=sample_rate,
                      stft_point=stft_point, duration_time=duration_time, fps=fps, file_type=file_type)

    def ShowSpectrogram(self,
                        data_path: str,
                        drone_name: str = 'test',
                        sample_rate: Union[int, float] = 100e6,
                        stft_point: int = 2048,
                        duration_time: float = 0.1,
                        oneside: bool = False,
                        Middle_Frequency: float = 2400e6,
                        file_type=np.float32
                        ):
        """tool used to observe the spectrograms from a local datapack
        :param save_path: the target dir path to save the image result.
        :param data_path: the input raw data file path.
        :param sample_rate: the simple rate when collect the raw frequency data using USRP, ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
        :param stft_point: the STFT points using in STFT transformation, you better set this to 2**n (n is a Non-negative integer). ref: https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT
        :param duration_time: the duration time of single spectromgram.
        :param oneside: set 'True' if you want to observe the real & imaginary parts separately, set 'False' to show the complete spectrogram
        :param Middle_Frequency: the middle frequency set to collect data using USRP in the frequency band, ref: https://en.wikipedia.org/wiki/Center_frequency
        """

        if oneside:
            show_half_only(datapack=data_path, drone_name=drone_name,
                           fs=sample_rate, stft_point=stft_point, duration_time=duration_time, file_type=file_type)

        else:
            show_spectrum(datapack=data_path, drone_name=drone_name, fs=sample_rate, stft_point=stft_point,
                           duration_time=duration_time, Middle_Frequency=Middle_Frequency, file_type=file_type)


def generate_images(datapack: str = None,
                    file: str = None,
                    pack: str = None,
                    fs: int = 100e6,
                    stft_point: int = 1024,
                    duration_time: float = 0.1,
                    ratio: int = 1,  # 控制产生图片时间间隔的倍率，默认为1生成视频的倍率
                    location: str = 'buffer',
                    file_type=np.float32
                    ):
    """
    Generates images from the given data using Short-Time Fourier Transform (STFT).

    Parameters:
    - datapack (str): Path to the data file.
    - file (str): File name.
    - pack (str): Pack name.
    - fs (int): Sampling frequency, default is 100 MHz. ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
    - stft_point (int): Number of points for STFT, default is 1024. ref: https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    - ratio (int): Controls the time interval ratio for generating images, default is 1.
    - location (str): Location to save the images, default is 'buffer'.

    Returns:
    - list: List of images if `location` is 'buffer'.
    """
    slice_point = int(fs * duration_time)
    data = np.fromfile(datapack, dtype=file_type)
    data = data[::2] + data[1::2] * 1j
    if location == 'buffer': images = []

    i = 0
    while (i + 1) * slice_point <= len(data):

        f, t, Zxx = STFT(data[int(i * slice_point): int((i + 1) * slice_point)],
                         stft_point=stft_point, fs=fs, duration_time=duration_time, onside=False)
        f = np.fft.fftshift(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)
        aug = 10 * np.log10(np.abs(Zxx))
        extent = [t.min(), t.max(), f.min(), f.max()]

        plt.figure()
        plt.imshow(aug, extent=extent, aspect='auto', origin='lower', cmap='jet')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)

        if location == 'buffer':
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300)
            plt.close()

            buffer.seek(0)
            images.append(Image.open(buffer))

        else:
            plt.savefig(location + (file + '/' if file else '') + (pack + '/' if pack else '') + file + ' (' + str(i) + ').jpg', dpi=300)
            plt.close()

        i += 2 ** (-ratio)

    if location == 'buffer':
        return images


def save_as_video(datapack: str,
                  save_path: str,
                  fs: int = 100e6,
                  stft_point: int = 1024,
                  duration_time: float = 0.1,
                  fps: int = 5,  # 视频帧率
                  file_type=np.float32
                  ):
    """
    Saves the generated images as a video.

    Parameters:
    - datapack (str): Path to the data file.
    - save_path (str): Path to save the video.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 1024.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    - fps (int): Frame rate of the video, default is 5.
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(datapack):
        raise ValueError('File not found!')

    images = generate_images(datapack=datapack, fs=fs, stft_point=stft_point, duration_time=duration_time, file_type=file_type)
    imageio.mimsave(save_path+'video.mp4', images, fps=fps)


def show_spectrum(datapack: str = '',
                  drone_name: str = 'test',
                  fs: int = 100e6,
                  stft_point: int = 2048,
                  duration_time: float = 0.1,
                  Middle_Frequency: float = 2400e6,
                  file_type=np.float32
                  ):

    """
    Displays the spectrum of the given data.

    Parameters:
    - datapack (str): Path to the data file.
    - drone_name (str): Name of the drone, default is 'test'.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 2048.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    - Middle_Frequency (float): Middle frequency, default is 2400 MHz.
    """

    with open(datapack, 'rb') as fp:
        print("reading raw data...")
        read_data = np.fromfile(fp, dtype=file_type)

        data = read_data[::2] + read_data[1::2] * 1j
        print('STFT transforming')

        f, t, Zxx = STFT(data, stft_point=stft_point, fs=fs, duration_time=duration_time, onside=False)
        f = np.linspace(Middle_Frequency-fs / 2, Middle_Frequency+fs / 2, stft_point)
        Zxx = np.fft.fftshift(Zxx, axes=0)

        plt.figure()
        aug = 10 * np.log10(np.abs(Zxx))
        extent = [t.min(), t.max(), f.min(), f.max()]
        plt.imshow(aug, extent=extent, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(drone_name)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()


def show_half_only(datapack: str = '',
                   drone_name: str = 'test',
                   fs: int = 100e6,
                   stft_point: int = 2048,
                   duration_time: float = 0.1,
                   file_type=np.float32
                   ):

    """
    Displays I and Q components of the given data separately.

    Parameters:
    - datapack (str): Path to the data file.
    - drone_name (str): Name of the drone, default is 'test'.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 2048.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    """

    with open(datapack, 'rb') as fp:
        print("reading raw data...")
        read_data = np.fromfile(fp, dtype=file_type)
        dataI = read_data[::2]
        dataQ = read_data[1::2]

        f_I, t_I, Zxx_I = STFT(dataI, fs=fs, stft_point=stft_point, duration_time=duration_time)
        f_Q, t_Q, Zxx_Q = STFT(dataQ, fs=fs, stft_point=stft_point, duration_time=duration_time)

        # I部分數據的時頻圖
        print('Drawing')
        plt.figure()
        aug_I = 10 * np.log10(np.abs(Zxx_I))
        plt.pcolormesh(t_I, f_I, np.abs(aug_I))
        plt.title(drone_name + " I")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
        plt.show()
        print("figure I done")

        # Q部分數據的時頻圖
        plt.figure()
        aug_Q = 10 * np.log10(np.abs(Zxx_Q))
        plt.pcolormesh(t_Q, f_Q, np.abs(aug_Q), cmap='jet')
        plt.title(drone_name + " Q")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
        plt.show()
        print("figure Q done")


def DrawandSave(
        fig_save_path: str,
        file_path: str,
        fs: int = 100e6,
        stft_point: int = 2048,
        duration_time: float = 0.1,
        file_type=np.float32
):

    """
    Draw and save the images from the given data files.

    Parameters:
    - fig_save_path (str): Path to save the figures.
    - file_path (str): Path to the data files.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 2048.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.

    Your raw data should organize like this:
    file_path
        Drone 1
            data pack1.iq
            data pack2.iq
            ...
            data packn.iq
        Drone 2
            data pack1.iq
            data pack2.iq
            ...
            data packn.iq
        Drone 3
            data pack1.iq
            data pack2.iq
            ...
            data packn.iq
        .....
        Drone n
            ...
    """
    re_files = os.listdir(file_path)

    for file in re_files:
        packlist = os.listdir(os.path.join(file_path, file))
        for pack in packlist:
            check_folder(os.path.join(fig_save_path, file, pack))
            generate_images(datapack=os.path.join(file_path, file, pack),
                            file=file,
                            pack=pack,
                            fs=fs,
                            stft_point=stft_point,
                            duration_time=duration_time,
                            ratio=0,
                            location=fig_save_path,
                            file_type=file_type
                            )

            print(pack + ' Done')
        print(file + ' Done')
    print('All Done')


def check_folder(folder_path):
    """
    Checks and creates the folder if it does not exist.

    Parameters:
    - folder_path (str): Path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"folder '{folder_path}' created。")
    else:
        print(f"folder '{folder_path}' existed。")


def STFT(data,
         onside: bool = True,
         stft_point: int = 1024,
         fs: int = 100e6,
         duration_time: float = 0.1,
         ):

    """
    Performs Short-Time Fourier Transform (STFT) on the given data.

    Parameters:
    - data (array-like): Input data.
    - onside (bool): Whether to return one-sided or two-sided STFT, default is True.
    - stft_point (int): Number of points for STFT, default is 1024.
    - fs (int): Sampling frequency, default is 100 MHz.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.

    Returns:
    - f (array): Frequencies.
    - t (array): Times.
    - Zxx (array): STFT result.
    """

    slice_point = int(fs * duration_time)

    f, t, Zxx = stft(data[0: slice_point], fs,
         return_onesided=onside, window=windows.hamming(stft_point), nperseg=stft_point)

    return f, t, Zxx


def _compute_fft_frame(data_segment, fft_size, window, frame_idx, fft_cache, cache_lock):
    """
    计算单个帧的FFT，支持数据复用缓存。
    
    Parameters:
    - data_segment: 数据段
    - fft_size: FFT窗口大小
    - window: 窗函数
    - frame_idx: 帧索引
    - fft_cache: FFT结果缓存字典
    - cache_lock: 缓存锁
    
    Returns:
    - frame_idx: 帧索引
    - magnitude: FFT幅度谱
    """
    # 检查缓存
    cache_key = (frame_idx, fft_size)
    with cache_lock:
        if cache_key in fft_cache:
            return frame_idx, fft_cache[cache_key]
    
    # 计算FFT
    frame_data = data_segment * window
    magnitude = np.abs(np.fft.fftshift(fft(frame_data)))
    
    # 存入缓存
    with cache_lock:
        fft_cache[cache_key] = magnitude
    
    return frame_idx, magnitude


def _generate_spectrogram_image(spectrogram_data, fs, fft_size, num_frames, frame_id):
    """
    生成时频图（颜色映射）。
    
    Parameters:
    - spectrogram_data: 频谱数据
    - fs: 采样率
    - fft_size: FFT大小
    - num_frames: 总帧数
    - frame_id: 帧ID
    
    Returns:
    - frame_id: 帧ID
    - image_buffer: 图像缓冲区
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(np.log10(spectrogram_data.T), aspect='auto', cmap='jet', origin='lower',
               extent=[0, num_frames * (fft_size) / fs, -fs / 2, fs / 2])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
    plt.axis('off')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    buffer.seek(0)
    
    return frame_id, BytesIO(buffer.getvalue())


def waterfall_spectrogram_optimized(datapack, fft_size, fs, location, time_scale, 
                                     target_frame_gap=150, num_workers=4):
    """
    优化的瀑布图生成函数，支持数据复用、并行处理和不阻塞流水线。

    Parameters:
    - datapack: 原始数据路径或numpy数组
    - fft_size: FFT窗口大小
    - fs: 采样率
    - location: 保存位置，'buffer'表示内存，否则为文件路径
    - time_scale: 时间尺度，控制何时开始滚动
    - target_frame_gap: 目标帧之间的间隔（小帧数）
    - num_workers: 并行工作线程数

    Returns:
    - images: 图像列表（当location='buffer'时）或None
    """
    # 加载整段原始信号
    if isinstance(datapack, str):
        data = np.fromfile(datapack, dtype=np.float32)
        data = data[::2] + data[1::2] * 1j
    elif isinstance(datapack, np.ndarray):
        data = datapack
    else:
        raise ValueError("datapack must be str or np.ndarray")
    
    window = np.hanning(fft_size)
    num_small_frames = len(data) // fft_size
    
    # 计算目标帧（视频帧）的切分点
    target_frame_indices = []
    if num_small_frames > time_scale:
        # 第一个目标帧在time_scale处
        target_frame_indices.append(time_scale)
        # 后续目标帧每隔target_frame_gap个小帧
        current = time_scale + target_frame_gap
        while current < num_small_frames:
            target_frame_indices.append(current)
            current += target_frame_gap
    
    if not target_frame_indices:
        return [] if location == 'buffer' else None
    
    # 创建保存目录
    if location != 'buffer':
        if not os.path.exists(location):
            os.makedirs(location)
        images = None
    else:
        images = []
    
    # FFT结果缓存（带时间顺序的哈希表）
    fft_cache = {}
    fft_cache_lock = Lock()
    fft_results = OrderedDict()  # 维护时间顺序
    
    # 时频图结果缓存（带时间顺序的哈希表）
    spectrogram_images = OrderedDict()
    spectrogram_lock = Lock()
    
    # 并行处理FFT
    def process_fft_for_target_frame(target_frame_idx):
        """为目标帧计算所需的FFT结果"""
        # 计算该目标帧需要的所有小帧的FFT
        start_frame = max(0, target_frame_idx - time_scale)
        end_frame = target_frame_idx + 1
        
        frame_ffts = []
        for small_frame_idx in range(start_frame, end_frame):
            data_segment = data[small_frame_idx * fft_size: (small_frame_idx + 1) * fft_size]
            _, magnitude = _compute_fft_frame(data_segment, fft_size, window, 
                                             small_frame_idx, fft_cache, fft_cache_lock)
            frame_ffts.append(magnitude)
        
        # 组装成频谱图
        spectrogram = np.array(frame_ffts)
        return target_frame_idx, spectrogram
    
    # 使用线程池并行处理所有目标帧的FFT
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        fft_futures = {executor.submit(process_fft_for_target_frame, idx): idx 
                      for idx in target_frame_indices}
        
        # 收集FFT结果
        for future in as_completed(fft_futures):
            target_frame_idx, spectrogram = future.result()
            fft_results[target_frame_idx] = spectrogram
    
    # 并行生成时频图（颜色映射）
    def generate_image_for_frame(target_frame_idx):
        """为目标帧生成时频图"""
        spectrogram = fft_results[target_frame_idx]
        _, image_buffer = _generate_spectrogram_image(
            spectrogram, fs, fft_size, num_small_frames, target_frame_idx
        )
        return target_frame_idx, image_buffer
    
    # 并行生成所有时频图
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        image_futures = {executor.submit(generate_image_for_frame, idx): idx 
                        for idx in target_frame_indices}
        
        # 收集时频图结果（按完成顺序，但保持时间顺序）
        for future in as_completed(image_futures):
            target_frame_idx, image_buffer = future.result()
            with spectrogram_lock:
                spectrogram_images[target_frame_idx] = image_buffer
    
    # 按时间顺序重组结果
    if location == 'buffer':
        images = [spectrogram_images[idx] for idx in sorted(spectrogram_images.keys())]
        return images
    else:
        for j, idx in enumerate(sorted(spectrogram_images.keys())):
            image_buffer = spectrogram_images[idx]
            image_buffer.seek(0)
            img = Image.open(image_buffer)
            img.save(os.path.join(location, f'{j}_waterfall_spectrogram.jpg'), 'JPEG', quality=95)
        return None


def waterfall_spectrogram(datapack, fft_size, fs, location, time_scale):
    """
    生成瀑布图（保持向后兼容的接口，内部调用优化版本）。
    
    Parameters:
    - datapack: 原始数据路径或numpy数组
    - fft_size: FFT窗口大小
    - fs: 采样率
    - location: 保存位置
    - time_scale: 时间尺度
    
    Returns:
    - images: 图像列表或None
    """
    return waterfall_spectrogram_optimized(datapack, fft_size, fs, location, time_scale)


# Usage-----------------------------------------------------------------------------------------------------------------
def main():

    """
    data_path = data_path
    save_path = save_path
    test = RawDataProcessor()
    test.TransRawDataintoSpectrogram(fig_save_path=save_path,
                                     data_path=data_path,
                                     sample_rate=100e6,
                                     stft_point=1024,
                                     duration_time=0.1,
                                     )
    """

    """
    data_path = ''
    test.ShowSpectrogram(data_path=data_path,
                         drone_name='DJ FPV COMBO',
                         sample_rate=100e6,
                         stft_point=2048,
                         duration_time=0.1,
                         Middle_Frequency=2400e6
                         )
    """

    """
    save_path = ''
    data_path = ''
    test.TransRawDataintoSpectrogram(fig_save_path=save_path,
                                 data_path=data_path,
                                 sample_rate=100e6,
                                 stft_point=2048,
                                 duration_time=0.1,
                                 )
    """

    """
    datapack = ''
    save_path = ''
    save_as_video(datapack=datapack,
                  save_path=save_path,
                  fs=100e6,
                  stft_point=1024,
                  duration_time=0.1,
                  fps=5,
                  )

    show_spectrum(datapack=datapack,
                  drone_name='test',
                  fs=100e6,
                  stft_point=2048,
                  duration_time=0.1,
                  )
    show_half_only(datapack, 
                   drone_name='test',
                   fs=100e6,
                   stft_point=2048,
                   duration_time=0.1,
                   )
    """


if __name__ == '__main__':
    main()