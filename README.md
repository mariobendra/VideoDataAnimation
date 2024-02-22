# VideoDataAnimation

VideoDataAnimation is a Python library designed for creating synchronized visualizations of video data and corresponding time-series data. It allows users to visualize changes in data alongside video frames, making it ideal for applications in research, data analysis, and educational purposes where visual context is crucial.

## Features

- **Synchronized Visualization**: Seamlessly integrate video playback with real-time data plots, ensuring synchronized visualizations that enhance understanding and analysis.
- **Interactive Plotting**: Utilize interactive plotting features to zoom, pan, or hover over data points for detailed inspection, providing a deeper analysis of the data alongside video content.
- **Flexible Data Integration**: Easily integrate various data sources by supporting multiple data formats, including CSV, Excel, and direct data frames, facilitating hassle-free data visualization.
- **Region of Interest Cropping**: Focus on specific areas within your videos by defining regions of interest, allowing for detailed analysis of targeted video segments.
- **Customizable Visual Styles**: Personalize your visualizations with customizable plot styles, including color schemes, line styles, and marker options, to match your presentation or branding requirements.
- **Dynamic Windowing**: Adjust the data viewing window dynamically, either by specifying a fixed number of data points or by setting a time window, to focus on specific segments of your data over time.
- **Annotation and Labeling**: Enhance your visualizations with annotations and labels, providing context and insights directly on your plots and video frames, making complex data more accessible.
- **Multiple Export Formats**: Export your synchronized video and data visualizations to a variety of formats such as MP4, AVI, GIF, or even as interactive HTML files, ensuring compatibility across different platforms and devices.
- **Batch Processing Support**: Automate the processing of multiple video and data pairs with batch processing capabilities, saving time and ensuring consistency across large datasets.
- **Extensive Documentation and Examples**: Get up and running quickly with comprehensive documentation, including detailed setup instructions, usage examples, and troubleshooting tips.
- **Community and Support**: Join an active community of users and contributors for support, to share ideas, and to collaborate on new features, making VideoDataAnimation not just a tool but a growing ecosystem.


## Installation

To install VideoDataAnimation, simply use pip:

pip install VideoDataAnimation

## Quick Start

    from VideoDataAnimation import VideoDataAnimation

**Initialize the VideoDataAnimation with your video and CSV file paths**··
    
    vda = VideoDataAnimation(
    csv_path='./comp_APP.csv',
    video_path='./comp_APP.avi',
    labels=['$m_{x}$', '$m_{y}$', '$m_{z}$'],
    crop_region=(145, 300, 1000, 400),
    window_size=None)

**Load the data, set up video capture, and prepare the plot**
    
    vda.load_data()
    vda.setup_video_capture()
    vda.setup_plot()

**Save the animation to an MP4 file, adjusting the playback speed with the slow_factor**··

    vda.save_animation('mp4', slow_factor=2)

**Release resources after saving the animation**··

    vda.release_resources()

For questions or support, please contact m.bendra22@gmail.com
