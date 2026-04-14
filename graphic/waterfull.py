from io import BytesIO
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft

from graphic.io_utils import _is_complex_dtype, _count_complex_samples, _complex_from_file


def plot_waterfall_spectrogram(iq_data=None,
                               datapack: str = None,
                               fs: int = 100e6,
                               output_dir: str = 'waterfall_output',
                               frame_size: int = 256,
                               plot_size: int = 512,
                               gap: int = 150,
                               file_type=np.float32,
                               chunk_size: int = 10_000_000,
                               ):
    """Plot a colored waterfall spectrogram and save each frame as an image.

    Supports two data sources:
    - Pre-loaded array via `iq_data` (np.ndarray).
    - Raw IQ file via `datapack` (str), loaded in streaming fashion.

    :param iq_data: IQ signal data as np.ndarray (complex form). Mutually exclusive with datapack.
    :param datapack: Path to raw IQ file. Mutually exclusive with iq_data.
    :param fs: Sampling frequency in Hz.
    :param output_dir: 'buffer' to return images in RAM, or a directory path to save to disk.
    :param frame_size: FFT window size, default 256.
    :param plot_size: Number of FFT frames in each waterfall column, default 512.
    :param gap: Frame step between consecutive waterfall outputs, default 150.
    :param file_type: NumPy dtype of the raw file, default np.float32.
    :param chunk_size: Max complex samples held in RAM per chunk (streaming mode), default 10M.
    :return: List of BytesIO buffers when output_dir='buffer', otherwise None.
    """
    if (iq_data is None) == (datapack is None):
        raise ValueError("Provide either iq_data (np.ndarray) or datapack (str), not both.")

    if output_dir != 'buffer':
        os.makedirs(output_dir, exist_ok=True)
        images = None
    else:
        images = []

    window = np.hanning(frame_size)
    spectrogram = []
    pack_gap = 0
    j = 0

    def _process_frame(start_idx):
        frame_data = iq_data_local[start_idx:start_idx + frame_size] * window
        magnitude = np.abs(np.fft.fftshift(fft(frame_data)))
        return magnitude

    def _save_waterfall_frame():
        nonlocal j, pack_gap
        spectrogram_arr = np.array(spectrogram)
        plt.figure()
        plt.imshow(np.log10(spectrogram_arr.T), aspect='auto', cmap='jet', origin='lower',
                   extent=[0, num_frames * frame_size / fs, -fs / 2, fs / 2])
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        plt.axis('off')

        if output_dir != 'buffer':
            plt.savefig(os.path.join(output_dir, f'{j}_waterfall_spectrogram.jpg'), dpi=300)
            plt.close()
        else:
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            plt.close()
            buf.seek(0)
            images.append(BytesIO(buf.getvalue()))

        j += 1
        pack_gap = 0

    # --- streaming mode: read file chunk by chunk ---
    if datapack is not None:
        total_complex = _count_complex_samples(datapack, file_type)
        num_frames = total_complex // frame_size

        chunk_start = 0
        while chunk_start < total_complex:
            chunk_end = min(chunk_start + chunk_size, total_complex)
            iq_data_local = _complex_from_file(datapack, chunk_start, chunk_end - chunk_start, file_type)
            chunk_frames = len(iq_data_local) // frame_size

            global_offset = chunk_start // frame_size

            for i in range(chunk_frames):
                global_i = global_offset + i
                magnitude = _process_frame(i * frame_size)

                if global_i > plot_size:
                    spectrogram = spectrogram[1:]
                    spectrogram = np.concatenate((spectrogram, magnitude.reshape(1, frame_size)), axis=0)
                elif global_i == plot_size:
                    spectrogram.append(magnitude)
                else:
                    spectrogram.append(magnitude)

                pack_gap += 1

                if pack_gap == gap:
                    _save_waterfall_frame()

            chunk_start = chunk_end

        # flush remaining spectrogram if it has enough frames
        if pack_gap > 0 and len(spectrogram) >= plot_size:
            _save_waterfall_frame()

        return images

    # --- in-memory mode: iq_data is already loaded ---
    iq_data_local = iq_data
    num_frames = len(iq_data) // frame_size

    for i in range(num_frames):
        magnitude = _process_frame(i * frame_size)

        if i > plot_size:
            spectrogram = spectrogram[1:]
            spectrogram = np.concatenate((spectrogram, magnitude.reshape(1, frame_size)), axis=0)
        elif i == plot_size:
            spectrogram.append(magnitude)
        else:
            spectrogram.append(magnitude)

        pack_gap += 1

        if pack_gap == gap:
            _save_waterfall_frame()

    if pack_gap > 0 and len(spectrogram) >= plot_size:
        _save_waterfall_frame()

    return images


def video_maker(image_folder: str = '',
                output_path: str = '',
                video_name: str = 'output.mp4',
                fps: int = 30):
    """Create a video from a series of images in a folder.

    :param image_folder: Path to the folder containing the images.
    :param output_path: Directory where the output video will be saved.
    :param video_name: Name of the output video file.
    :param fps: Frame rate of the output video, default 30.
    :return: Path to the created video file, or None if creation failed.
    """
    if not os.path.isdir(image_folder):
        print(f"Error: image folder does not exist: {image_folder}")
        return None

    images = sorted([img for img in os.listdir(image_folder)
                     if img.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not images:
        print(f"Error: no images found in {image_folder}")
        return None

    first_img = cv2.imread(os.path.join(image_folder, images[0]))
    if first_img is None:
        print(f"Error: failed to read first image: {images[0]}")
        return None

    height, width = first_img.shape[:2]
    os.makedirs(output_path, exist_ok=True)
    video_path = os.path.join(output_path, video_name)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not video.isOpened():
        print(f"Error: VideoWriter failed to open {video_path}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return None

    for img_name in images:
        frame = cv2.imread(os.path.join(image_folder, img_name))
        if frame is None:
            print(f"Warning: skipped unreadable image: {img_name}")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    return video_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Waterfall spectrogram tool')
    parser.add_argument('--data', '-d', required=True, help='Path to IQ data file (.iq / .bin)')
    parser.add_argument('--output', '-o', default='waterfall_output',
                        help='Output directory (default: waterfall_output)')
    parser.add_argument('--fs', type=float, default=100e6,
                        help='Sampling frequency in Hz (default: 100e6)')
    parser.add_argument('--frame-size', type=int, default=256,
                        help='FFT window size (default: 256)')
    parser.add_argument('--plot-size', type=int, default=512,
                        help='FFT frames per waterfall column (default: 512)')
    parser.add_argument('--gap', type=int, default=150,
                        help='Frame gap between waterfall outputs (default: 150)')
    parser.add_argument('--video', action='store_true',
                        help='Also produce a video from saved images')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frame rate (default: 30)')
    parser.add_argument('--dtype', default='float32',
                        choices=['float32', 'float64', 'complex64'],
                        help='Data file type (default: float32)')
    parser.add_argument('--chunk-size', type=int, default=10_000_000,
                        help='Max complex samples per chunk for streaming (default: 10M)')

    args = parser.parse_args()

    file_type_map = {'float32': np.float32, 'float64': np.float64, 'complex64': np.complex64}
    dtype = file_type_map[args.dtype]

    print(f"Loading data from {args.data} ...")
    plot_waterfall_spectrogram(
        datapack=args.data,
        fs=int(args.fs),
        output_dir=args.output,
        frame_size=args.frame_size,
        plot_size=args.plot_size,
        gap=args.gap,
        file_type=dtype,
        chunk_size=args.chunk_size,
    )
    print(f"Images saved: {args.output}")

    if args.video:
        print("Creating video ...")
        video_path = video_maker(
            image_folder=args.output,
            output_path=args.output,
            video_name='waterfall.mp4',
            fps=args.fps,
        )
        if video_path:
            print(f"Video ready: {video_path}")


if __name__ == "__main__":
    main()
