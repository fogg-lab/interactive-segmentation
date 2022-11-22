# Interactive Semantic Segmentation

Built on top of the following repositories:  
- https://github.com/XavierCHEN34/ClickSEG
- https://github.com/saic-vul/ritm_interactive_segmentation

The labeling application runs on Linux, Windows, and Mac.
<br></br>
## Labeling application setup  
### Prerequisites
If you pick installation option 2, first install a Conda distribution like [Miniconda](https://docs.conda.io/en/latest/miniconda.html). During the installation process for Miniconda (or Anaconda), check the box to add conda to your `path` environmental variable.

### Installation Option 1 (Automatic)
1. Clone this repository, or download and extract [the zip file](https://github.com/fogg-lab/interactive-segmentation/archive/refs/heads/main.zip) to a folder.  
**Additional step for Mac or Linux users**: 
    - Right-click on the project folder (the folder that contains `environment.yml` and this README file), and select `Open in terminal`, or `New terminal at folder`. If neither of these options exist, you can do this instead:  
      - Right-click on the project folder and copy the path
      - Open a terminal (or command prompt) and navigate to the project folder by entering `cd <path to project folder>`, e.g. `cd /home/user/interactive-segmentation`
    - In the terminal, enter a command to add the execute permission to the install and launch scripts: `chmod u+x install_scripts/iseg-mac-installer.command && chmod u+x launch_scripts/iseg-mac-launcher.command`
#
2. Double click on the installation script for your operating system to install the labeling application:  
    - Windows: `install_scripts/iseg-windows-installer.bat` (if you get a security warning, click `More info` and then `Run anyway`)  
    - Mac: `install_scripts/iseg-mac-installer.command`  
    - Linux: `install_scripts/install_iseg.sh` (execute it from the command line)

### Installation Option 2 (Manual)
1. Clone this repository, or download and extract [the zip file](https://github.com/fogg-lab/interactive-segmentation/archive/refs/heads/main.zip) to a folder.  
Right-click on the project folder (the folder that contains `environment.yml` and this README file) and copy the path for the next step.
#
2. In a terminal or command prompt, navigate to the `interactive-segmentation` install folder with `cd` and the path you copied in the previous step.  
For example:  
    > `cd C:\Users\username\Downloads\interactive-segmentation`  
#
3. Use conda to install the environment for running the labeling app.  
    > `conda env create -f environment.yml python=3.10`  
#
4. Download a checkpoint file (.pth) for a trained interactive segmentation model, and move it into the checkpoints folder (e.g. `/Users/username/interactive-segmentation/checkpoints/`).  
You can download a trained model (trained for endothelial tube network segmentation) at [this link](https://drive.google.com/file/d/1JJZalxTMQFL9grnEBmHNQ37IezOhjDYZ/view?usp=share_link).
<br></br>
### Usage
1. (Launch using the launch script) Double click the appropriate launch script for your operating system, located in the `launch_scripts` directory.

&nbsp;&nbsp;&nbsp;&nbsp;OR

1. (Launch using the command line) Launch the application from the command line after activating the `iseg` conda environment created in step 3 of the manual installation instructions.  
From the terminal or command prompt, activate the Conda environment, navigate to the project folder, and launch the labeling app.  
For example:  
    > `cd /Users/username/repositories/interactive-segmentation`  
    > `conda activate iseg`  
    > `python iseg.py`
#
2. Load an image file in the labeling app, and optionally load an existing segmentation mask as well.
   Press the `Load image` button on the top bar to load an image file from your computer.  
   *If you have an existing mask you want to refine or continue working on, press the `Load mask` button to load it.*
#
3. Use these controls to label the image:  
    - Scroll with your mouse to zoom in/out, and right click + drag to pan around in the image.
    - Click on an object to select it for segmentation, or right click on the background to omit it from the segmentation.  
    - Use the `Toggle brush` button to switch between brush and click modes. The brush mode lets you paint on foreground (positive) and background (negative) selections. The `Erase brushstrokes` option erases both foreground and background (invisible) brushstrokes, independent from the click selection layer.  
    - Use the `Selection Transparency` (alpha blending coefficient) slider to change the transparency of the selection layer.  
    - Press `Show/Hide Mask` to show or hide the resulting segmentation mask.
#
4. Save frequently to avoid losing your work.

**Note**: If needed, configuration variables can be changed in the `config.yml` file.
<br></br>
## Demo (walkthrough of features and example usage)  
![Demo](./assets/img/demo.gif)


## Training a segmentation model
For now, please see the documentation for these other repositories for more information:
- https://github.com/XavierCHEN34/ClickSEG
- https://github.com/saic-vul/ritm_interactive_segmentation
