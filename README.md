# Interactive Semantic Segmentation

Built on top of the following repositories:  
- https://github.com/XavierCHEN34/ClickSEG
- https://github.com/saic-vul/ritm_interactive_segmentation

<br>

**Note:** We are currently working on packaging the project into a standalone program that can be distributed to any platform, without the need to manually install any dependencies or launch the app from the command line.

<br>

### Setup to run the application
1. Open a terminal and clone the repository  
```
    git clone git@github.com:fogg-lab/interactive-segmentation.git
    cd interactive-segmentation
```

2. Install the requirements
```
    pip install -r requirements.txt
```  
or  
```
    conda env create -f environment.yml
```  
3. Get a checkpoint file (.pth) for a trained interactive segmentation model, and change the value of the configuration variable "checkpoint-path" in config.yml to the filepath (i.e C:/Users/Bob/my_model_checkpoint.pth). You can download a trained model (trained for endothelial tube network segmentation) [here](https://drive.google.com/file/d/1JJZalxTMQFL9grnEBmHNQ37IezOhjDYZ/view?usp=share_link).  

4. Specify other configuration parameters in `config.yml` (or use the defaults)  

5. Launch the application  
```
    python demo.py
```

<br>

### Further documentation in progress...
