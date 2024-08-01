# Download weights
```commandline
sh_utils
sh get_weights_safety_ris.sh
```

# Docker setup
```commandline
cd docker
sh build.sh
sh start.sh
```

# Environment
## create dataset
```commandline
python utilite_cross_dataset.py
```

# Validation
## dataset 1
```commandline
sh validate_safety_ris_dataset_1.sh
```
## dataset 2
```commandline
sh validate_safety_ris_dataset_2.sh
```

# Train
```commandline
sh train_safety_ris.sh
```
