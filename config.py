from box import Box

config = {
    "num_devices": 2,
    "batch_size": 12,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint": "/home/nehaprakriya/sam_training/lightning-sam/sam_vit_h_4b8939.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/home/nehaprakriya/sam_training/data/fashionpedia/train",
            "annotation_file": "/home/nehaprakriya/sam_training/data/fashionpedia/instances_attributes_train2020.json"
        },
        "val": {
            "root_dir": "/home/nehaprakriya/sam_training/data/fashionpedia/test",
            "annotation_file": "/home/nehaprakriya/sam_training/data/fashionpedia/instances_attributes_val2020.json"
        }
    }
}

cfg = Box(config)
