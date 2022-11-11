import yaml
import subprocess

n_workers = 32
batch_size = 8

"""
Dataset directory structure:
    dataset_root
        train
            images
                <image name>_image.<extension>
            masks
                <image name>_mask.<extension>
        val
            images
                <image name>_image.<extension>
            masks
                <image name>_mask.<extension>
"""

with open("training_config.yml", "r", encoding="utf-8") as cfg_file:
    config = yaml.safe_load(cfg_file)

# Training config variables
dataset_path = config['dataset_path']
train_script = config['train_script']
model_path = config['model_path']
pretrained_weights = config['pretrained_weights']
exp_name = config['exp_name']

# Run training
train_args = [
    model_path,
    f"--pretrained_weights={pretrained_weights}",
    f"--dataset_path={dataset_path}",
    "--gpus=0",
    #f"--ngpus={n_gpus}",
    f"--workers={n_workers}",
    f"--batch-size={batch_size}",
    f"--exp-name={exp_name}",
    #"--resume-exp=000",
    #"--resume-prefix=42",
    #"--start-epoch=43"
]

execute_training_cmd = f"python3 {train_script} {' '.join(train_args)}"

subprocess.check_output(execute_training_cmd, shell=True)
