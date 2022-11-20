# Interactive Semantic Segmentation

Built on top of the following repositories:  
- https://github.com/XavierCHEN34/ClickSEG
- https://github.com/saic-vul/ritm_interactive_segmentation

<br>

## Labeling application setup
### Prerequisites
Install a Conda distribution like [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  
The labeling application runs on Linux, Windows, and Mac. This program's user interface can be less responsive on Mac, so Windows or Linux is preferred.

### Installation
1. Clone this repository or download and extract [the zip file](https://github.com/fogg-lab/interactive-segmentation/archive/refs/heads/main.zip) to a folder.  
Copy the full path to the install folder (e.g. `/home/username/repositories/interactive-segmentation/install`) for the next step.

2. In a terminal or command prompt, navigate to the `interactive-segmentation` install folder with `cd` and the path you copied in the previous step.  
For example:  
    > `cd C:\Users\username\Downloads\interactive-segmentation-main\install`  

3. Pick the matching environment.yml file for your system, and use Conda to install the environment for running the labeling app.  
**Note**: If you are running Windows or Linux and do not have NVIDIA GPU, use an environment file ending in `_cpu.yml` instead of `_gpu.yml`.
For example:
    > `conda env create -f environment_windows_cpu.yml`  
Or for Mac OS:  
    > `conda env create -f environment_mac.yml`

4. *Important*: Download a checkpoint file (.pth) for a trained interactive segmentation model, and move it into the checkpoints folder (e.g. `/Users/username/interactive-segmentation/checkpoints/`).  
You can download a trained model (trained for endothelial tube network segmentation) at [this link](https://drive.google.com/file/d/1JJZalxTMQFL9grnEBmHNQ37IezOhjDYZ/view?usp=share_link).  

### Usage  
1. Launch the application from the command line after activating the `iseg` conda environment created in step 3 of the installation instructions.  
From the terminal or command prompt, activate the Conda environment, navigate to the project folder, and launch the labeling app.  
For example:  
> `cd /Users/username/repositories/interactive-segmentation`  
> `conda activate iseg`  
> `python iseg.py`

2. Load an image file in the labeling app, and optionally load an existing segmentation mask as well.
   Press the `Load image` button on the top bar to load an image file from your computer.  
   *If you have an existing mask you want to refine or continue working on, press the `Load mask` button to load it.*

3. Click on an object to select it for segmentation, or right click on the background to omit it from the segmentation.  
   Use the `Toggle brush` button to switch between brush and click modes. The brush mode lets you paint on foreground (positive) and background (negative) selections.  
   For example:  
   

4. Save frequently to avoid losing your work. This is expressly recommended because the application is still in early development, and it could crash.  

**Note**: If needed, configuration variables can be changed in the `config.yml` file.
