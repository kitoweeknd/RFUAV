from io import BytesIO
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft


def plot_waterfall_spectrogram(iq_data,
                               fs: int,
                               output_dir: str,
                               frame_size: int = 256,
                               plot_size: int = 39062,
                               ):

    """
    Plot a colored waterfall spectrogram and save each frame as an image.

    :param iq_data: IQ signal data (complex form)
    :param frame_size: Size of each frame
    :param fs: Sampling frequency
    :param output_dir: Directory to save images
    :param plot_size: Number of frames to include in the initial plot
    """

    if output_dir != 'buffer':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        images = []

    num_frames = len(iq_data) // frame_size
    window = np.hanning(frame_size)
    spectrogram = []
    pack_gap = 0
    j = 0
    gap = 150

    for i in range(num_frames):
        start_idx = i * (frame_size)
        magnitude = np.abs(np.fft.fftshift(fft(iq_data[start_idx:start_idx + frame_size] * window)))
        if i > plot_size:
            spectrogram = spectrogram[1:]
            spectrogram = np.concatenate((spectrogram, magnitude.reshape(1, frame_size)), axis=0)

        else:
            spectrogram.append(magnitude)

        pack_gap += 1

        if i == plot_size:
            spectrogram = np.array(spectrogram)
            plt.figure()
            plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                       extent=[0, num_frames * (frame_size) / fs, -fs / 2, fs / 2])
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
            plt.axis('off')

            if output_dir != 'buffer':
                plt.savefig(os.path.join(output_dir, str(j) + 'waterfall_spectrogram.jpg'), dpi=300)
                plt.close()

            else:
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300)
                plt.close()
                buffer.seek(0)
                images.append(BytesIO(buffer.getvalue()))

            j += 1
            pack_gap = 0

        if i > plot_size and pack_gap == gap:
            plt.figure()
            plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                       extent=[0, num_frames * (frame_size) / fs, -fs / 2, fs / 2])
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
            if output_dir != 'buffer':
                plt.savefig(os.path.join(output_dir, str(j) + 'waterfall_spectrogram.jpg'), dpi=300)
                plt.close()
            else:
                plt.savefig(buffer, format='png', dpi=300)
                plt.close()
                buffer.seek(0)
                images.append(BytesIO(buffer.getvalue()))
            j += 1
            pack_gap = 0


def video_maker(image_folder: str = '',
                output_path: str = '',
                video_name: str = ''):

    """
    Creates a video from a series of images located in a specified folder.

    Parameters:
    image_folder (str): The path to the folder containing the images.
    output_path (str): The path where the output video will be saved.
    video_name (str): The name of the output video file.

    Returns:
    None
    """

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


# Usage-----------------------------------------------------------------------------------------------------------------
def main():

    # load data
    save_path = ''
    data_path = ''
    data = np.fromfile(data_path, dtype=np.float32)
    data = data[::2] + data[1::2] * 1j

    """
    datapack = ''
    data = h5py.File(datapack, 'r')
    data_I = data['RF0_I'][0]
    data_Q = data['RF0_Q'][0]
    data = data_I + data_Q * 1j
    """

    plot_waterfall_spectrogram(iq_data=data, output_dir=save_path)


if __name__ == "__main__":
    main()