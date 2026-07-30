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
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import stft, windows
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Union
from scipy.fft import fft
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

from graphic.io_utils import _count_complex_samples, _complex_from_file


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
                     window=windows.hamming(stft_point).astype(np.float32),
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
                    file_type=np.float32,
                    dpi: int = None
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
    - dpi (int): Output resolution. Defaults to 300 (unchanged from before) when None,
        so a bare Figure (6.4×4.8 in) renders at 1920×1440 exactly as the old path did.

    Returns:
    - list of PIL Images (when location='buffer'), otherwise None.
    """
    slice_point = int(fs * duration_time)
    total_complex = _count_complex_samples(datapack, file_type)
    n_frames = int(total_complex // slice_point)

    dpi_eff = 300 if dpi is None else dpi
    images = [] if location == 'buffer' else None
    frame_count = 0

    # Build the Figure / Agg canvas / Axes ONCE and reuse them across frames,
    # replacing the per-frame global `plt.figure()` (slow + not thread-safe) with
    # the object-oriented API (same pattern as `_generate_spectrogram_image`). A
    # bare Figure() inherits the rcParams default figsize (6.4×4.8 in), so at
    # dpi=300 the PNG stays 1920×1440 — identical appearance to the old path.
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    # Rasterise the reused canvas at `dpi_eff` so a bare Figure (6.4×4.8 in) draws
    # at 1920×1440 for the default dpi=300 — matching the old savefig(dpi=300)
    # pixels — and so the buffer path below can read the Agg buffer directly.
    fig.set_dpi(dpi_eff)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = None

    for frame_idx in range(0, n_frames, step):
        # Stream one frame from disk (only ~1.6 MB for float32 at 100 MHz, 0.1 s)
        data = _complex_from_file(datapack, frame_idx * slice_point, slice_point, file_type)

        f, t, Zxx = STFT(data, stft_point=stft_point, fs=fs,
                         duration_time=duration_time, onside=False)
        f = np.fft.fftshift(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)
        aug = 10 * np.log10(np.abs(Zxx))
        aug = np.clip(aug, a_min=1e-10, a_max=None)

        if im is None:
            # `extent` is fixed across frames (depends only on fs/stft_point/
            # slice_point), so the AxesImage is created just once.
            extent = [t.min(), t.max(), f.min(), f.max()]
            im = ax.imshow(aug, extent=extent, aspect='auto',
                           origin='lower', cmap='jet')
        else:
            im.set_data(aug)
            im.autoscale()  # re-derive clim per frame, matching per-frame imshow

        if location == 'buffer':
            # Read the RGBA raster straight off the Agg canvas instead of encoding
            # a PNG and decoding it back: identical pixels (same Agg raster at
            # dpi_eff, same RGBA mode the old Image.open(png) produced), but the
            # PNG compress+decompress round-trip is gone. np.array() COPIES the
            # buffer, so the next frame's canvas.draw() cannot corrupt this image
            # (the guard the old lazy Image.open().load() used to provide).
            canvas.draw()
            images.append(Image.fromarray(np.array(canvas.buffer_rgba())))
        else:
            subdir = os.path.join(location,
                                  (file + '/' if file else '') +
                                  (pack + '/' if pack else ''))
            os.makedirs(subdir, exist_ok=True)
            fname = os.path.join(subdir, f'{file} ({frame_count}).jpg')
            fig.savefig(fname, dpi=dpi_eff)

        frame_count += 1

    return images


def frames_to_video(frames, video_path: str, fps: int = 5):
    """Encode an ordered sequence of frames into an H.264 / yuv420p mp4 by piping
    raw RGB video straight into the system ffmpeg over stdin.

    ``frames`` is any iterable of PIL.Image (any mode) or HxW(x3|x4) uint8 arrays;
    each is coerced to contiguous RGB and ffmpeg is sized to the first one. This is
    the single video encoder shared by ``save_as_video`` and the classify raw-data
    pipeline, so neither needs imageio's ffmpeg backend — only the ffmpeg *binary*
    the image installs (docker/Dockerfile). Output matches the old files: libx264 /
    yuv420p / crf 23.

    stderr is kept tiny (``-loglevel error -nostats``) and fully drained before
    ``wait()``, so a large frame count can't fill the pipe and deadlock the writer;
    stdin is closed exactly once (never via ``communicate()``, which would re-flush
    the already-closed handle -> "ValueError: flush of closed file").
    """
    import subprocess

    proc = None
    n = 0
    try:
        for frame in frames:
            if isinstance(frame, np.ndarray):
                arr = frame
            else:  # PIL.Image (possibly RGBA / palette) -> RGB
                arr = np.asarray(frame.convert('RGB'))
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
            h, w = arr.shape[:2]

            if proc is None:
                # Sized to the first frame. yuv420p needs even dimensions; the pad
                # filter is a no-op at the 640×480 default and only rescues odd
                # sizes from a custom dpi.
                proc = subprocess.Popen([
                    'ffmpeg', '-y', '-loglevel', 'error', '-nostats',
                    '-f', 'rawvideo',
                    '-pix_fmt', 'rgb24',
                    '-s', f'{w}x{h}',
                    '-framerate', str(fps),
                    '-i', '-',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '23',
                    video_path,
                ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)

            proc.stdin.write(arr.tobytes())
            n += 1

        if proc is None:
            print('No frames to encode (empty input).')
            return

        proc.stdin.close()
        stderr = proc.stderr.read()
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, 'ffmpeg', stderr=stderr)
        print(f'Video ready: {video_path}  ({n} frames @ {fps} fps)')

    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode() if e.stderr else e}")
        raise
    finally:
        # If we bailed mid-stream, don't leave ffmpeg running.
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass
            proc.kill()
            proc.wait()


def save_as_video(datapack: str,
                  save_path: str,
                  fs: int = 100e6,
                  stft_point: int = 1024,
                  duration_time: float = 0.1,
                  fps: int = 5,
                  file_type=np.float32,
                  dpi: int = None
                  ):
    """Save the raw data as an MP4 video, streaming frames straight into ffmpeg.

    Each spectrogram frame is rendered on a single reused Figure (object-oriented
    Agg canvas — no per-frame global pyplot) and piped as raw RGB into a long-lived
    ffmpeg process over stdin. This removes both the PNG-to-disk round-trip and the
    per-frame figure allocation, while the output stays byte-for-byte the same kind
    of file (libx264 / yuv420p / crf 23 at `save_path/video.mp4`).

    Parameters:
    - datapack (str): Path to the raw IQ data file.
    - save_path (str): Directory in which to save video.mp4.
    - fs (int): Sampling frequency in Hz, default 100 MHz.
    - stft_point (int): FFT window size, default 1024.
    - duration_time (float): Duration per frame in seconds, default 0.1.
    - fps (int): Output video frame rate, default 5.
    - file_type: NumPy dtype of the raw file, default np.float32.
    - dpi (int): Frame resolution. None keeps the previous appearance — a bare
        Figure at the default 100 dpi renders 640×480, exactly as before.
    """
    os.makedirs(save_path, exist_ok=True)
    video_path = os.path.join(save_path, 'video.mp4')

    slice_point = int(fs * duration_time)
    total_complex = _count_complex_samples(datapack, file_type)
    n_frames = int(total_complex // slice_point)

    # Reused OO figure/canvas. Frame pixel size = figsize × figure dpi; the default
    # (6.4×4.8 in @ 100 dpi) yields 640×480, matching the old savefig(format='png').
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    if dpi is not None:
        fig.set_dpi(dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    def render_frames():
        im = None
        for i in range(n_frames):
            data = _complex_from_file(datapack, i * slice_point, slice_point, file_type)

            f, t, Zxx = STFT(data, stft_point=stft_point, fs=fs,
                             duration_time=duration_time, onside=False)
            f = np.fft.fftshift(f)
            Zxx = np.fft.fftshift(Zxx, axes=0)
            aug = 10 * np.log10(np.abs(Zxx))
            aug = np.clip(aug, a_min=1e-10, a_max=None)

            if im is None:
                extent = [t.min(), t.max(), f.min(), f.max()]
                im = ax.imshow(aug, extent=extent, aspect='auto',
                               origin='lower', cmap='jet')
            else:
                # autoscale() must run every frame: the old per-frame plt.imshow
                # picked clim from each frame's own min/max, so skipping it would
                # shift the colour mapping.
                im.set_data(aug)
                im.autoscale()

            canvas.draw()
            # buffer_rgba() -> HxWx4; frames_to_video drops alpha and pipes RGB.
            yield np.asarray(canvas.buffer_rgba())

            if (i + 1) % 100 == 0 or i == n_frames - 1:
                print(f"  frame {i + 1}/{n_frames}", flush=True)

    print("Encoding video with ffmpeg (stdin pipe) ...")
    frames_to_video(render_frames(), video_path, fps=fps)


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
    data = _complex_from_file(datapack, 0, slice_point, file_type)
    dataI = data.real
    dataQ = data.imag

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


# ----------------------------------------------------------------------
# Waterfall spectrogram (batched FFT + streaming)
# ----------------------------------------------------------------------
def _generate_spectrogram_image(spectrogram_data, fs, fft_size, num_frames, frame_id):
    """Render one waterfall spectrogram to a PNG buffer.

    Uses the matplotlib object-oriented API (Figure + Agg canvas) rather than the
    global pyplot state, so it is safe to call concurrently from worker threads.

    Returns: (frame_id, BytesIO buffer)
    """
    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(np.log10(spectrogram_data.T),
              aspect='auto', cmap='jet', origin='lower',
              extent=[0, num_frames * fft_size / fs, -fs / 2, fs / 2])

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300)
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

    window = np.hanning(fft_size).astype(np.float32)

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

        # Each target only needs the `time_scale`-frame window ending at its own
        # frame: mags[start_frame : local_idx + 1]. The previous version FFT'd
        # EVERY frame in the chunk (num_small_frames — up to ~48k for a 50M chunk)
        # and then sliced these small windows out, which is wasteful whenever the
        # targets are sparse (target_frame_gap ≥ time_scale — the common case).
        # Here we FFT only the union of the needed windows: same per-frame input →
        # same magnitude rows → pixel-for-pixel identical images, but the FFT work
        # drops to what is actually rendered. When windows overlap heavily the
        # merged span approaches the whole chunk, so this is never worse than the
        # old all-frames pass. End-clamp to num_small_frames matches the previous
        # numpy slice, which silently truncated the tail window.
        fft_block = 20_000
        windows_needed = sorted((max(0, li - time_scale), min(li + 1, num_small_frames))
                                for li in local_indices)
        merged = []
        for s, e in windows_needed:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        mag_blocks = []  # (interval_start, interval_end, magnitudes) covering `merged`
        for s, e in merged:
            nf = e - s
            arr = np.empty((nf, fft_size), dtype=np.float32)
            for b in range(0, nf, fft_block):
                be = min(b + fft_block, nf)
                block = data[(s + b) * fft_size: (s + be) * fft_size].reshape(be - b, fft_size)
                arr[b:be] = np.abs(np.fft.fftshift(fft(block * window, axis=1), axes=1))
            mag_blocks.append((s, e, arr))

        def _window_mags(a, b):
            # `merged` covers every requested window by construction, so exactly
            # one block contains [a, b); return that slice (no copy).
            for s, e, arr in mag_blocks:
                if s <= a and b <= e:
                    return arr[a - s: b - s]
            raise RuntimeError(f"window [{a}, {b}) not covered by any FFT block")

        fft_results = OrderedDict()
        for local_idx, global_idx in zip(local_indices, target_frame_indices):
            start_frame = max(0, local_idx - time_scale)  # clamp to chunk boundary
            fft_results[global_idx] = _window_mags(start_frame,
                                                   min(local_idx + 1, num_small_frames))

        def generate_image_for_frame(global_idx):
            _, buf = _generate_spectrogram_image(
                fft_results[global_idx], fs, fft_size, num_small_frames, global_idx)
            return global_idx, buf

        # Rendering uses the thread-safe OO matplotlib API, so it can still fan
        # out; results are collected in the main thread (no lock needed).
        spectrogram_images = OrderedDict()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            img_futures = {executor.submit(generate_image_for_frame, gi): gi
                           for gi in target_frame_indices}
            for future in as_completed(img_futures):
                gi, buf = future.result()
                spectrogram_images[gi] = buf

        # --- write images to disk / collect in RAM ---
        if location == 'buffer':
            images_chunk = [spectrogram_images[gi] for gi in
                            sorted(spectrogram_images)]
            processed_frames += len(images_chunk)
            all_images.extend(images_chunk)
            print(f"  [{chunk_start:,} – {chunk_end:,}] "
                  f"{len(images_chunk)} waterfall frames  (total: {processed_frames})")
        else:
            for j, gi in enumerate(sorted(spectrogram_images)):
                buf = spectrogram_images[gi]
                buf.seek(0)
                # The buffer is a PNG (RGBA); JPEG has no alpha channel, so drop it
                # before saving (convert also forces the lazy load off the buffer).
                img = Image.open(buf).convert('RGB')
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
    """Backward-compatible alias for ``waterfall_spectrogram_optimized`` (the
    documented public entry point); delegates with default gap/workers/chunk."""
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
