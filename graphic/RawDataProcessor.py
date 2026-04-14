# Tool to process raw data
import matplotlib
matplotlib.use('Agg')
import argparse
import inspect
import os
import shutil
import sys
import tempfile
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Union
from scipy.fft import fft
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import OrderedDict

from graphic.io_utils import _is_complex_dtype, _count_complex_samples, _complex_from_file, _complex_memmap


# ----------------------------------------------------------------------
# RawDataProcessor
# ----------------------------------------------------------------------
class RawDataProcessor:
    """Transform raw data into images, video, and save the result locally.

    Modes:
    - trans_raw_data_into_spectrogram(): batch-convert raw data to spectrogram images
    - trans_raw_data_into_video(): convert raw data to video
    - show_spectrogram(): quick interactive spectrogram preview (first slice only)
    """

    def trans_raw_data_into_spectrogram(self,
                                        fig_save_path: str,
                                        data_path: str,
                                        sample_rate: Union[int, float] = 100e6,
                                        stft_point: int = 2048,
                                        duration_time: float = 0.1,
                                        file_type=np.float32
                                        ):
        """Transform the raw data into spectrograms and save the results locally.

        :param fig_save_path: the target dir path to save the image result.
        :param data_path: the input raw data dir path.
        :param sample_rate: the sampling rate used during USRP acquisition.
        :param stft_point: STFT window size; should be a power of 2.
        :param duration_time: duration of a single spectrogram frame in seconds.
        """
        draw_and_save(fig_save_path=fig_save_path, file_path=data_path, fs=sample_rate,
                      stft_point=stft_point, duration_time=duration_time, file_type=file_type)

    def trans_raw_data_into_video(self,
                                  save_path: str,
                                  data_path: str,
                                  sample_rate: Union[int, float] = 100e6,
                                  stft_point: int = 2048,
                                  duration_time: float = 0.1,
                                  fps: int = 5,
                                  file_type=np.float32
                                  ):
        """Transform the raw data into video and save the result locally.

        :param save_path: the target dir path to save the video.
        :param data_path: the input raw data file path.
        :param sample_rate: the sampling rate used during USRP acquisition.
        :param stft_point: STFT window size; should be a power of 2.
        :param duration_time: duration of a single spectrogram frame in seconds.
        :param fps: frame rate of the output video.
        """
        save_as_video(datapack=data_path, save_path=save_path, fs=sample_rate,
                      stft_point=stft_point, duration_time=duration_time, fps=fps,
                      file_type=file_type)

    def show_spectrogram(self,
                        data_path: str,
                        drone_name: str = 'test',
                        sample_rate: Union[int, float] = 100e6,
                        stft_point: int = 2048,
                        duration_time: float = 0.1,
                        oneside: bool = False,
                        file_type=np.float32,
                        save_path: str = None
                        ):
        """Interactive preview of the first `duration_time` seconds of the raw data.

        Falls back to saving the figure when no GUI is available.

        :param data_path: the input raw data file path.
        :param sample_rate: the sampling rate used during USRP acquisition.
        :param stft_point: STFT window size; should be a power of 2.
        :param duration_time: duration of the displayed spectrogram segment in seconds.
        :param oneside: True = show I and Q components separately; False = full spectrogram.
        :param file_type: NumPy dtype of the raw file.
        :param save_path: if provided, always save to this path instead of displaying.
        """
        if oneside:
            show_half_only(datapack=data_path, drone_name=drone_name,
                           fs=sample_rate, stft_point=stft_point,
                           duration_time=duration_time, file_type=file_type,
                           save_path=save_path)
        else:
            show_spectrum(datapack=data_path, drone_name=drone_name, fs=sample_rate,
                          stft_point=stft_point, duration_time=duration_time,
                          file_type=file_type, save_path=save_path)


# ----------------------------------------------------------------------
# Core processing functions
# ----------------------------------------------------------------------
def STFT(data,
         onside: bool = True,
         stft_point: int = 1024,
         fs: int = 100e6,
         duration_time: float = 0.1,
         ):
    """Perform Short-Time Fourier Transform on the given data.

    Parameters:
    - data (array-like): Input complex data.
    - onside (bool): Whether to return one-sided STFT, default True.
    - stft_point (int): FFT window size, default 1024.
    - fs (int): Sampling frequency in Hz, default 100 MHz.
    - duration_time (float): Duration of each segment in seconds, default 0.1.

    Returns: (frequencies, times, Zxx)
    """
    slice_point = int(fs * duration_time)
    f, t, Zxx = stft(data[0: slice_point], fs,
                     return_onesided=onside,
                     window=windows.hamming(stft_point),
                     nperseg=stft_point)
    return f, t, Zxx


def generate_images(datapack: str = None,
                    file: str = None,
                    pack: str = None,
                    fs: int = 100e6,
                    stft_point: int = 1024,
                    duration_time: float = 0.1,
                    step: int = 1,
                    location: str = 'buffer',
                    file_type=np.float32
                    ):
    """Generate spectrogram images from a raw data file using streaming reads.

    When `location` is 'buffer', returns a list of PIL Images.
    When `location` is a directory path, saves each image to disk immediately
    (memory-efficient: only one frame is held in RAM at a time).

    Parameters:
    - datapack (str): Path to the raw IQ data file.
    - file (str): Top-level label (e.g., drone type), used in filenames.
    - pack (str): Sub-label (e.g., recording session), used in filenames.
    - fs (int): Sampling frequency in Hz, default 100 MHz.
    - stft_point (int): FFT window size, default 1024.
    - duration_time (float): Duration per frame in seconds, default 0.1.
    - step (int): Only keep every N-th frame (step=1 means every frame, step=2 means every
        other frame, etc.), default 1.
    - location (str): 'buffer' to return images in RAM, or a directory path to stream to disk.
    - file_type: NumPy dtype of the raw file, default np.float32.

    Returns:
    - list of PIL Images (when location='buffer'), otherwise None.
    """
    slice_point = int(fs * duration_time)
    total_complex = _count_complex_samples(datapack, file_type)
    n_frames = int(total_complex // slice_point)

    images = [] if location == 'buffer' else None
    frame_count = 0

    for frame_idx in range(0, n_frames, step):
        # Stream one frame from disk (only ~1.6 MB for float32 at 100 MHz, 0.1 s)
        data = _complex_from_file(datapack, frame_idx * slice_point, slice_point, file_type)

        f, t, Zxx = STFT(data, stft_point=stft_point, fs=fs,
                         duration_time=duration_time, onside=False)
        f = np.fft.fftshift(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)
        aug = np.clip(aug, a_min=1e-10, a_max=None)
        extent = [t.min(), t.max(), f.min(), f.max()]

        plt.figure()
        plt.imshow(aug, extent=extent, aspect='auto', origin='lower', cmap='jet')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)

        if location == 'buffer':
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            plt.close()
            buf.seek(0)
            images.append(Image.open(buf))
        else:
            subdir = os.path.join(location,
                                  (file + '/' if file else '') +
                                  (pack + '/' if pack else ''))
            os.makedirs(subdir, exist_ok=True)
            fname = os.path.join(subdir, f'{file} ({frame_count}).jpg')
            plt.savefig(fname, dpi=300)
            plt.close()

        frame_count += 1

    return images


def save_as_video(datapack: str,
                  save_path: str,
                  fs: int = 100e6,
                  stft_point: int = 1024,
                  duration_time: float = 0.1,
                  fps: int = 5,
                  file_type=np.float32
                  ):
    """Save the raw data as an MP4 video, streaming frames to avoid memory overflow.

    Frames are written as PNG files in a temp directory (one at a time, never all in RAM)
    and encoded to MP4 via ffmpeg at the end.

    Parameters:
    - datapack (str): Path to the raw IQ data file.
    - save_path (str): Directory in which to save video.mp4.
    - fs (int): Sampling frequency in Hz, default 100 MHz.
    - stft_point (int): FFT window size, default 1024.
    - duration_time (float): Duration per frame in seconds, default 0.1.
    - fps (int): Output video frame rate, default 5.
    - file_type: NumPy dtype of the raw file, default np.float32.
    """
    import subprocess

    os.makedirs(save_path, exist_ok=True)
    video_path = os.path.join(save_path, 'video.mp4')

    slice_point = int(fs * duration_time)
    total_complex = _count_complex_samples(datapack, file_type)
    n_frames = int(total_complex // slice_point)

    tmp_dir = tempfile.mkdtemp(prefix='rdu_video_')

    try:
        for i in range(n_frames):
            data = _complex_from_file(datapack, i * slice_point, slice_point, file_type)

            f, t, Zxx = STFT(data, stft_point=stft_point, fs=fs,
                             duration_time=duration_time, onside=False)
            f = np.fft.fftshift(f)
            Zxx = np.fft.fftshift(Zxx, axes=0)
            aug = 10 * np.log10(np.abs(Zxx))
            aug = np.clip(aug, a_min=1e-10, a_max=None)
            extent = [t.min(), t.max(), f.min(), f.max()]

            plt.figure()
            plt.imshow(aug, extent=extent, aspect='auto', origin='lower', cmap='jet')
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)

            frame_path = os.path.join(tmp_dir, f'frame_{i:06d}.png')
            plt.savefig(frame_path, format='png')
            plt.close()

            if (i + 1) % 100 == 0 or i == n_frames - 1:
                print(f"  frame {i + 1}/{n_frames}", flush=True)

        print("Encoding video with ffmpeg ...")
        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(tmp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            video_path
        ], check=True, capture_output=True)

        print(f'Video ready: {video_path}  ({n_frames} frames @ {fps} fps)')

    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode() if e.stderr else e}")
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _is_interactive_backend() -> bool:
    """Return True if the current matplotlib backend supports plt.show()."""
    backend = matplotlib.get_backend().lower()
    # Known non-interactive (file-only) backends
    non_interactive = {'agg', 'svg', 'pdf', 'ps', 'cairo', 'none'}
    return backend not in non_interactive


def _save_figure(fig, name: str) -> str:
    """Save figure to a temp file and return its path.

    Saved alongside the script so users can find it easily even when
    the temp dir is obscure (e.g. /tmp on Linux vs %TEMP% on Windows).
    """
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(_save_figure)))
    safe_name = os.path.splitext(os.path.basename(name))[0]
    out_path = os.path.join(script_dir, f'{safe_name}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def show_spectrum(datapack: str = '',
                  drone_name: str = 'test',
                  fs: int = 100e6,
                  stft_point: int = 2048,
                  duration_time: float = 0.1,
                  file_type=np.float32,
                  save_path: str = None
                  ):
    """Display or save the spectrogram of the first `duration_time` seconds of the file.

    When no GUI is available (Agg backend) the figure is saved automatically.

    Parameters:
    - datapack (str): Path to the raw IQ data file.
    - drone_name (str): Title for the plot, default 'test'.
    - fs (int): Sampling frequency in Hz, default 100 MHz.
    - stft_point (int): FFT window size, default 2048.
    - duration_time (float): Duration to display in seconds, default 0.1.
    - file_type: NumPy dtype of the raw file, default np.float32.
    - save_path (str): If provided, always save to this path; otherwise auto-save
        when no display is available.
    """
    slice_point = int(fs * duration_time)

    print("reading raw data ...")
    data = _complex_from_file(datapack, 0, slice_point, file_type)
    print("STFT transforming ...")

    f, t, Zxx = STFT(data, stft_point=stft_point, fs=fs,
                     duration_time=duration_time, onside=False)
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    fig = plt.figure()
    aug = 10 * np.log10(np.abs(Zxx))
    aug = np.clip(aug, a_min=1e-10, a_max=None)
    extent = [t.min(), t.max(), f.min(), f.max()]
    plt.imshow(aug, extent=extent, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(drone_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    elif not _is_interactive_backend():
        saved = _save_figure(fig, drone_name)
        print(f"[headless] Saved: {saved}")
    else:
        plt.show()


def show_half_only(datapack: str = '',
                   drone_name: str = 'test',
                   fs: int = 100e6,
                   stft_point: int = 2048,
                   duration_time: float = 0.1,
                   file_type=np.float32,
                   save_path: str = None
                   ):
    """Display or save I and Q components of the first `duration_time` seconds separately.

    When no GUI is available the figures are saved automatically.

    Parameters:
    - datapack (str): Path to the raw IQ data file.
    - drone_name (str): Title base for the plots, default 'test'.
    - fs (int): Sampling frequency in Hz, default 100 MHz.
    - stft_point (int): FFT window size, default 2048.
    - duration_time (float): Duration to display in seconds, default 0.1.
    - file_type: NumPy dtype of the raw file, default np.float32.
    - save_path (str): If provided, always save to this path; otherwise auto-save
        when no display is available.
    """
    slice_point = int(fs * duration_time)

    print("reading raw data ...")
    bytes_per_sample = np.dtype(file_type).itemsize
    raw = np.fromfile(datapack, dtype=file_type, offset=0,
                      count=slice_point * 2)
    dataI = raw[::2]
    dataQ = raw[1::2]

    print("STFT transforming ...")
    f_I, t_I, Zxx_I = STFT(dataI, fs=fs, stft_point=stft_point, duration_time=duration_time)
    f_Q, t_Q, Zxx_Q = STFT(dataQ, fs=fs, stft_point=stft_point, duration_time=duration_time)

    headless = not _is_interactive_backend()

    # I component
    fig_I = plt.figure()
    aug_I = 10 * np.log10(np.abs(Zxx_I))
    aug_I = np.clip(aug_I, a_min=1e-10, a_max=None)
    plt.pcolormesh(t_I, f_I, aug_I, cmap='jet')
    plt.title(drone_name + " I")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()

    if save_path:
        base, ext = os.path.splitext(save_path)
        path_I = f'{base}_I{ext}'
        fig_I.savefig(path_I, dpi=150, bbox_inches='tight')
        plt.close(fig_I)
        print(f"Saved: {path_I}")
    elif headless:
        saved_I = _save_figure(fig_I, f'{drone_name}_I')
        print(f"[headless] Saved I: {saved_I}")
    else:
        plt.show()

    # Q component
    fig_Q = plt.figure()
    aug_Q = 10 * np.log10(np.abs(Zxx_Q))
    aug_Q = np.clip(aug_Q, a_min=1e-10, a_max=None)
    plt.pcolormesh(t_Q, f_Q, aug_Q, cmap='jet')
    plt.title(drone_name + " Q")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()

    if save_path:
        base, ext = os.path.splitext(save_path)
        path_Q = f'{base}_Q{ext}'
        fig_Q.savefig(path_Q, dpi=150, bbox_inches='tight')
        plt.close(fig_Q)
        print(f"Saved: {path_Q}")
    elif headless:
        saved_Q = _save_figure(fig_Q, f'{drone_name}_Q')
        print(f"[headless] Saved Q: {saved_Q}")
    else:
        plt.show()


def draw_and_save(fig_save_path: str,
                file_path: str,
                fs: int = 100e6,
                stft_point: int = 2048,
                duration_time: float = 0.1,
                file_type=np.float32
                ):
    """Batch-generate spectrogram images from a directory tree of raw IQ files.

    Expected directory layout:
        file_path/
            Drone1/
                datapack1.iq
                datapack2.iq
                ...
            Drone2/
                ...

    Parameters:
    - fig_save_path (str): Root directory for saved images.
    - file_path (str): Root directory of the raw data tree.
    - fs (int): Sampling frequency in Hz, default 100 MHz.
    - stft_point (int): FFT window size, default 2048.
    - duration_time (float): Duration per frame in seconds, default 0.1.
    - file_type: NumPy dtype of the raw files, default np.float32.
    """
    re_files = os.listdir(file_path)

    for file in re_files:
        packlist = os.listdir(os.path.join(file_path, file))
        for pack in packlist:
            src = os.path.join(file_path, file, pack)
            dst = os.path.join(fig_save_path, file, pack)
            os.makedirs(dst, exist_ok=True)
            generate_images(datapack=src,
                            file=file,
                            pack=pack,
                            fs=fs,
                            stft_point=stft_point,
                            duration_time=duration_time,
                            step=1,
                            location=fig_save_path,
                            file_type=file_type
                            )
            print(f"  {pack} ... done")
        print(f"  {file} ... done")
    print("All done.")


def check_folder(folder_path):
    """Create `folder_path` if it does not exist (no-op if it already exists)."""
    os.makedirs(folder_path, exist_ok=True)


# ----------------------------------------------------------------------
# Waterfall spectrogram (data-reuse + parallel + streaming)
# ----------------------------------------------------------------------
def _compute_fft_frame(data_segment, fft_size, window, frame_idx, fft_cache, cache_lock):
    """Compute FFT for a single frame with cache.

    Returns: (frame_idx, magnitude)
    """
    cache_key = (frame_idx, fft_size)
    with cache_lock:
        if cache_key in fft_cache:
            return frame_idx, fft_cache[cache_key]

    frame_data = data_segment * window
    magnitude = np.abs(np.fft.fftshift(fft(frame_data)))

    with cache_lock:
        fft_cache[cache_key] = magnitude

    return frame_idx, magnitude


def _generate_spectrogram_image(spectrogram_data, fs, fft_size, num_frames, frame_id):
    """Generate a spectrogram image from the given data.

    Returns: (frame_id, BytesIO buffer)
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(np.log10(spectrogram_data.T), aspect='auto', cmap='jet', origin='lower',
               extent=[0, num_frames * fft_size / fs, -fs / 2, fs / 2])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    return frame_id, BytesIO(buf.getvalue())


def waterfall_spectrogram_optimized(datapack,
                                    fft_size: int,
                                    fs: int,
                                    location: str,
                                    time_scale: int,
                                    target_frame_gap: int = 150,
                                    num_workers: int = None,
                                    file_type=np.float32,
                                    chunk_size: int = 50_000_000
                                    ):
    """Generate waterfall spectrograms with parallel FFT and streaming data loading.

    Memory usage is bounded to `chunk_size` complex samples (~400 MB for float32,
    ~800 MB for float64) regardless of file size.

    Parameters:
    - datapack: Path to the raw IQ file, or a pre-loaded np.ndarray.
    - fft_size (int): FFT window size.
    - fs (int): Sampling frequency in Hz.
    - location (str): 'buffer' to return images in RAM, or a directory to stream to disk.
    - time_scale (int): Number of FFT frames visible in each waterfall column.
    - target_frame_gap (int): Spacing between consecutive waterfall frames (in FFT frames).
    - num_workers (int): Number of parallel FFT workers. Defaults to CPU count.
    - file_type: NumPy dtype of the raw file, default np.float32.
    - chunk_size (int): Maximum complex samples held in RAM at once, default 50M
        (~400 MB for float32, ~800 MB for float64). Adjust based on available RAM.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    # --- load / validate data source ---
    if isinstance(datapack, str):
        total_complex = _count_complex_samples(datapack, file_type)
    elif isinstance(datapack, np.ndarray):
        total_complex = len(datapack)
    else:
        raise ValueError("datapack must be str or np.ndarray")

    if location != 'buffer':
        os.makedirs(location, exist_ok=True)

    window = np.hanning(fft_size)

    # --- streaming pass: process the file in chunks ---
    processed_frames = 0

    all_images = [] if location == 'buffer' else None

    for chunk_start in range(0, total_complex, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_complex)
        chunk_len = chunk_end - chunk_start

        if isinstance(datapack, str):
            data = _complex_from_file(datapack, chunk_start, chunk_len, file_type)
        else:
            data = datapack[chunk_start:chunk_end]

        num_small_frames = len(data) // fft_size
        if num_small_frames <= time_scale:
            continue

        # Determine which target frames fall inside this chunk
        global_start = chunk_start // fft_size
        global_end = (chunk_end - 1) // fft_size + 1

        target_frame_indices = []
        if global_start < time_scale:
            target_frame_indices.append(time_scale)
        current = time_scale + target_frame_gap
        while current < global_end:
            target_frame_indices.append(current)
            current += target_frame_gap

        target_frame_indices = [idx for idx in target_frame_indices
                                if idx >= global_start and idx < global_end]
        if not target_frame_indices:
            continue

        print(f"  processing [{chunk_start:,} – {chunk_end:,}] ...")

        # Local indices within this chunk
        local_indices = [idx - global_start for idx in target_frame_indices]

        fft_cache = {}
        fft_cache_lock = Lock()
        fft_results = OrderedDict()
        spectrogram_images = OrderedDict()
        spectrogram_lock = Lock()

        def process_fft_for_target_frame(local_idx, global_idx):
            start_frame = max(0, global_idx - time_scale)
            end_frame = global_idx + 1
            frame_ffts = []
            for sf_idx in range(start_frame, end_frame):
                data_seg = data[sf_idx * fft_size: (sf_idx + 1) * fft_size]
                _, magnitude = _compute_fft_frame(data_seg, fft_size, window,
                                                   sf_idx, fft_cache, fft_cache_lock)
                frame_ffts.append(magnitude)
            spectrogram = np.array(frame_ffts)
            return global_idx, spectrogram

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            fft_futures = {executor.submit(process_fft_for_target_frame, li, gi): gi
                           for li, gi in zip(local_indices, target_frame_indices)}
            for future in as_completed(fft_futures):
                gi, spectrogram = future.result()
                fft_results[gi] = spectrogram

        def generate_image_for_frame(global_idx):
            spectrogram = fft_results[global_idx]
            _, buf = _generate_spectrogram_image(
                spectrogram, fs, fft_size, num_small_frames, global_idx)
            return global_idx, buf

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            img_futures = {executor.submit(generate_image_for_frame, gi): gi
                           for gi in target_frame_indices}
            for future in as_completed(img_futures):
                gi, buf = future.result()
                with spectrogram_lock:
                    spectrogram_images[gi] = buf

        # --- write images to disk / collect in RAM ---
        if location == 'buffer':
            images_chunk = [spectrogram_images[gi] for gi in
                            sorted(spectrogram_images, key=lambda x: x)]
            processed_frames += len(images_chunk)
            all_images.extend(images_chunk)
            print(f"  [{chunk_start:,} – {chunk_end:,}] "
                  f"{len(images_chunk)} waterfall frames  (total: {processed_frames})")
        else:
            for j, gi in enumerate(sorted(spectrogram_images)):
                buf = spectrogram_images[gi]
                buf.seek(0)
                img = Image.open(buf)
                img.save(os.path.join(location,
                                      f'{processed_frames + j}_waterfall_spectrogram.jpg'),
                         'JPEG', quality=95)
                img.close()
            processed_frames += len(spectrogram_images)
            print(f"  [{chunk_start:,} – {chunk_end:,}] "
                  f"{len(spectrogram_images)} waterfall frames  (total: {processed_frames})")

    return all_images if location == 'buffer' else None


def waterfall_spectrogram(datapack, fft_size, fs, location, time_scale,
                          file_type=np.float32):
    """Backward-compatible wrapper. Calls waterfall_spectrogram_optimized."""
    return waterfall_spectrogram_optimized(
        datapack, fft_size, fs, location, time_scale, file_type=file_type)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='RawDataProcessor',
        description='Transform raw IQ data into spectrograms, video, or real-time visualization.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='mode', title='mode', required=True)

    # ── spectrogram ──────────────────────────────────────────────────
    p_spec = subparsers.add_parser('spectrogram', help='Batch-generate spectrogram images')
    p_spec.add_argument('--input', '-i', required=True,
                         help='Input raw data directory')
    p_spec.add_argument('--output', '-o', required=True,
                         help='Output image directory')
    p_spec.add_argument('--fs', type=float, default=100e6,
                         help='Sampling rate (default 100e6)')
    p_spec.add_argument('--stft-point', type=int, default=2048,
                         help='STFT window size (default 2048)')
    p_spec.add_argument('--duration', type=float, default=0.1,
                         help='Single-frame duration in seconds (default 0.1)')
    p_spec.add_argument('--dtype', default='float32',
                         choices=['float32', 'float64', 'complex64'],
                         help='Data file type (default float32)')

    # ── video ────────────────────────────────────────────────────────
    p_vid = subparsers.add_parser('video', help='Convert raw data into video')
    p_vid.add_argument('--input', '-i', required=True,
                        help='Input raw data file (.iq)')
    p_vid.add_argument('--output', '-o', required=True,
                        help='Output directory for video.mp4')
    p_vid.add_argument('--fs', type=float, default=100e6,
                        help='Sampling rate (default 100e6)')
    p_vid.add_argument('--stft-point', type=int, default=1024,
                        help='STFT window size (default 1024)')
    p_vid.add_argument('--duration', type=float, default=0.1,
                        help='Single-frame duration in seconds (default 0.1)')
    p_vid.add_argument('--fps', type=int, default=5,
                        help='Video frame rate (default 5)')
    p_vid.add_argument('--dtype', default='float32',
                        choices=['float32', 'float64', 'complex64'],
                        help='Data file type (default float32)')

    # ── show ─────────────────────────────────────────────────────────
    p_show = subparsers.add_parser('show', help='Interactively display spectrogram')
    p_show.add_argument('--input', '-i', required=True,
                         help='Input raw data file (.iq)')
    p_show.add_argument('--name', default='test',
                         help='Drone/sample name (default test)')
    p_show.add_argument('--fs', type=float, default=100e6,
                         help='Sampling rate (default 100e6)')
    p_show.add_argument('--stft-point', type=int, default=2048,
                         help='STFT window size (default 2048)')
    p_show.add_argument('--duration', type=float, default=0.1,
                         help='Single-frame duration in seconds (default 0.1)')
    p_show.add_argument('--half', action='store_true',
                         help='Show I/Q components separately (half spectrum)')
    p_show.add_argument('--save', '-s', default=None,
                         help='Save the figure to this path instead of displaying '
                              '(useful in headless environments)')
    p_show.add_argument('--dtype', default='float32',
                         choices=['float32', 'float64', 'complex64'],
                         help='Data file type (default float32)')

    # ── waterfall ───────────────────────────────────────────────────
    p_wf = subparsers.add_parser('waterfall', help='Generate scrolling spectrogram (waterfall)')
    p_wf.add_argument('--input', '-i', required=True,
                      help='Input raw data file (.iq)')
    p_wf.add_argument('--output', '-o', required=True,
                      help='Output image directory')
    p_wf.add_argument('--fs', type=float, default=100e6,
                      help='Sampling rate (default 100e6)')
    p_wf.add_argument('--fft-size', type=int, default=1024,
                      help='FFT window size (default 1024)')
    p_wf.add_argument('--time-scale', type=int, default=512,
                      help='Time-window size (default 512)')
    p_wf.add_argument('--frame-gap', type=int, default=150,
                      help='Target frame interval in FFT frames (default 150)')
    p_wf.add_argument('--workers', type=int, default=None,
                      help='Number of parallel workers (default: CPU count)')
    p_wf.add_argument('--chunk-size', type=int, default=50_000_000,
                      help='Max complex samples held in RAM at once (default 50M, ~400 MB)')
    p_wf.add_argument('--dtype', default='float32',
                      choices=['float32', 'float64', 'complex64'],
                      help='Data file type (default float32)')

    return parser


def _dtype_from_str(name: str):
    return {'float32': np.float32, 'float64': np.float64, 'complex64': np.complex64}[name]


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    dtype = _dtype_from_str(args.dtype)

    if args.mode == 'spectrogram':
        proc = RawDataProcessor()
        proc.trans_raw_data_into_spectrogram(
            fig_save_path=args.output,
            data_path=args.input,
            sample_rate=args.fs,
            stft_point=args.stft_point,
            duration_time=args.duration,
            file_type=dtype,
        )

    elif args.mode == 'video':
        save_as_video(
            datapack=args.input,
            save_path=args.output,
            fs=args.fs,
            stft_point=args.stft_point,
            duration_time=args.duration,
            fps=args.fps,
            file_type=dtype,
        )

    elif args.mode == 'show':
        if args.half:
            show_half_only(
                datapack=args.input,
                drone_name=args.name,
                fs=args.fs,
                stft_point=args.stft_point,
                duration_time=args.duration,
                file_type=dtype,
                save_path=args.save,
            )
        else:
            show_spectrum(
                datapack=args.input,
                drone_name=args.name,
                fs=args.fs,
                stft_point=args.stft_point,
                duration_time=args.duration,
                file_type=dtype,
                save_path=args.save,
            )

    elif args.mode == 'waterfall':
        waterfall_spectrogram_optimized(
            datapack=args.input,
            fft_size=args.fft_size,
            fs=args.fs,
            location=args.output,
            time_scale=args.time_scale,
            target_frame_gap=args.frame_gap,
            num_workers=args.workers,
            file_type=dtype,
            chunk_size=args.chunk_size,
        )
        print(f"Waterfall images saved: {args.output}")


if __name__ == '__main__':
    main()
