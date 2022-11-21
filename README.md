# Interactive Semantic Segmentation

Built on top of the following repositories:  
- https://github.com/XavierCHEN34/ClickSEG
- https://github.com/saic-vul/ritm_interactive_segmentation

The labeling application runs on Linux, Windows, and Mac.  

## Labeling application setup  
### Prerequisites
For installation option 2, install a Conda distribution like [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Installation Option 1 (Automatic)
1. Clone this repository, or download and extract [the zip file](https://github.com/fogg-lab/interactive-segmentation/archive/refs/heads/main.zip) to a folder.  
**Additional step for Mac or Linux users**: 
    - Right-click on the project folder (the folder that contains the `environment.yml` file) and select `Open in terminal`, or `New terminal at folder`. If neither of these options exist, you can do this instead:  
      - Right-click on the project folder (the folder that contains the `environment.yml` file) and copy the path
      - Open a terminal (or command prompt) and navigate to the project folder by entering `cd <path to project folder>`, e.g. `cd /home/user/interactive-segmentation`
    - In the terminal, enter a command to add the execute permission to the install and launch scripts: `chmod u+x install_scripts/iseg-mac.command && chmod u+x launch_scripts/iseg-mac-launcher.command`
#
2. Double click on the installation script for your operating system to install the labeling application:  
    - Windows: `install_scripts/iseg-win.bat` (if you get a security warning, click `More info` and then `Run anyway`)  
    - Mac: `install_scripts/iseg-mac.command`  
    - Linux: `install_scripts/iseg_linux.sh`

### Installation Option 2 (Manual)
1. Clone this repository or download and extract [the zip file](https://github.com/fogg-lab/interactive-segmentation/archive/refs/heads/main.zip) to a folder.  
Right-click on the project folder (the folder that contains the `environment.yml` file) and copy the path for the next step.
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

### Usage
1. (Launch using the launch script) Double click the appropriate launch script for your operating system, located in the `launch_scripts` directory.

OR

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
3. Click on an object to select it for segmentation, or right click on the background to omit it from the segmentation.  
   Use the `Toggle brush` button to switch between brush and click modes. The brush mode lets you paint on foreground (positive) and background (negative) selections. 
#
4. Save frequently to avoid losing your work.

<br>

### Demo (features overview and how to use the labeling app)  
![Demo](./assets/img/demo.gif)

**Note**: If needed, configuration variables can be changed in the `config.yml` file.
