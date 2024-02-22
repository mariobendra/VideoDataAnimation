import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import seaborn as sns
import pandas as pd
import tqdm
import warnings

try:
    import cv2
except ImportError:
    print("OpenCV (cv2) is not installed. Please install it using 'pip install opencv-python'.")

try:
    import seaborn as sns
except ImportError:
    print("seaborn is not installed. Please install it using 'pip install seaborn'.")

try:
    import tqdm
except ImportError:
    print("tqdm is not installed. Please install it using 'pip install tqdm'.")

# Suppress specific UserWarning about identical x-limits
warnings.filterwarnings("ignore", message="Attempting to set identical low and high xlims makes transformation singular; automatically expanding.")

class VideoDataAnimation:
    def __init__(self, csv_path, video_path, labels, crop_region=None, window_size=None):
        """
        Initialize the VideoDataAnimation object with configuration for video and data visualization.

        Parameters:
        - csv_path (str): Path to the CSV file containing data for animation.
        - video_path (str): Path to the video file for side-by-side visualization.
        - labels (list): List of labels for data columns in the CSV.
        - crop_region (tuple): Optional. Defines the region (x, y, width, height) to crop from the video.
        - window_size (int): Optional. Specifies the number of data points to display in the moving window.
        """
        self.setup_plot_style()
        self.csv_path = csv_path
        self.video_path = video_path
        self.crop_region = crop_region
        self.labels = labels
        self.data = None
        self.cap = None
        self.fps = 0
        self.total_video_frames = 0
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.window_size = window_size

    def setup_plot_style(self):
        """Configure the visual style of plots using seaborn and matplotlib settings."""
        sns.set_context("paper")
        sns.set_style("whitegrid")
        params = {
            "ytick.color": "black",
            "xtick.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "text.usetex": True,
            "font.family": "serif",
            "text.usetex": "True",
            "font.serif": ["Computer Modern Serif"]
        }
        plt.rcParams.update(params)

    def load_data(self):
        """Load CSV data for animation, and set labels for data columns."""
        try:
            self.data = pd.read_csv(self.csv_path)
            self.labels = self.data.columns[1:]
        except Exception as e:
            print(f"Failed to load data from {self.csv_path}: {e}")
            raise

    def setup_video_capture(self):
        """Set up video capture for the specified video file and extract key properties."""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video file at {self.video_path}.")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            print(f"Error setting up video capture: {e}")
            raise

    def setup_plot(self):
        """Initialize the plot with configurations for video and data visualization."""
        self.ax1.set_position([0.05, 0.1, 0.4, 0.8])
        self.ax2.set_xlabel("Time in [s]", fontsize=17)
        self.ax2.set_ylabel("Magnetization $m_{i}$", fontsize=17)
        self.ax2.set_ylim(-1, 1)

        line_styles = ['-', '--', '-.', ':']
        marker_styles = ['o', 's', '^', 'D', 'p']
        self.lines = []
        for label, (line_style, marker_style) in zip(self.labels, zip(line_styles, marker_styles)):
            line, = self.ax2.plot([], [], line_style, marker=marker_style, markevery=350, markersize=7, linewidth=1.5, label=label)
            self.lines.append(line)
        self.ax2.legend(loc='lower right')

    def init_animation(self):
        """Initialize animation by setting empty data for each plot line."""
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def animate(self, i):
        """
        Update function for the animation; called for each frame, with added progress bar in the terminal.

        Parameters:
        - i (int): Frame index in the animation sequence.
        """
        try:
            data_points_per_frame = len(self.data) / self.total_video_frames
            plot_index = int(i * data_points_per_frame)
            plot_index = min(plot_index, len(self.data) - 1)

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                return self.lines

            if self.crop_region:
                x, y, w, h = self.crop_region
                frame = frame[y:y + h, x:x + w]

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.ax1.clear()
            self.ax1.imshow(frame)
            self.ax1.axis('off')

            if self.window_size:
                start_index = max(0, plot_index - self.window_size)
                end_index = min(len(self.data), plot_index + self.window_size + 1)
            else:
                start_index = 0
                end_index = plot_index + 1

            for line, label in zip(self.lines, self.labels):
                line.set_data(self.data.iloc[start_index:end_index, 0], self.data[label][start_index:end_index])
                parent_ax = line.axes
                if self.window_size:
                    parent_ax.set_xlim(self.data.iloc[start_index, 0], self.data.iloc[end_index - 1, 0])
                else:
                    parent_ax.set_xlim(0, self.data.iloc[plot_index, 0])

                parent_ax.set_ylim(self.data[label].min(), self.data[label].max())
                self.ax2.set_ylim(-1, 1)

            return self.lines
        except Exception as e:
            print(f"Error during animation: {e}")
            return self.lines

    def save_animation(self, file_extension, slow_factor=1):
        """
        Save the generated animation as AVI, MP4, GIF, etc. file, with a progress bar showing the save status.

        Parameters:
        - file_extension: 'avi', 'mp4', 'gif', etc.
        - slow_factor (float): Factor to slow down the animation. Default is 1 (no slowing).
        """
        base_file_name = "animation_with_video"  # Base name for the file
        try:
            slowed_fps = self.fps / slow_factor
            interval = 1000 / slowed_fps

            # Construct the full file path
            file_path = f"{base_file_name}.{file_extension}"

            # Determine the appropriate writer based on the file extension and instantiate it with necessary parameters
            if file_extension == 'gif':
                writer = PillowWriter(fps=slowed_fps)
            elif file_extension in ['mp4', 'avi', 'mov']:
                writer = 'ffmpeg'  # You can create an instance of FFMpegWriter with specific options if needed
                writer = matplotlib.animation.FFMpegWriter(fps=slowed_fps)  # Example instantiation with fps
            else:
                raise ValueError(f"Unsupported file extension: .{file_extension}")

            # Create the animation with a progress bar
            anim = FuncAnimation(self.fig, self.animate, init_func=self.init_animation,
                                 frames=tqdm.tqdm(range(self.total_video_frames)),
                                 interval=interval, blit=True)

            # Save the animation to the specified file path using the instantiated writer
            anim.save(file_path, writer=writer)
            print(f"Animation saved to {file_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")

    def release_resources(self):
        """Release resources such as video capture objects and close the plot window."""
        if self.cap:
            self.cap.release()
        plt.close(self.fig)

# Due to the environment limitations, the following code is commented out and should be executed in an appropriate environment.
vda = VideoDataAnimation(csv_path='./comp_APP.csv', video_path='./comp_APP.avi', labels=['$m_{x}$', '$m_{y}$', '$m_{z}$'], crop_region=(145, 300, 1000, 400), window_size=None)
vda.load_data()
vda.setup_video_capture()
vda.setup_plot()
vda.save_animation('mp4', slow_factor=2)
vda.release_resources()
