from pathlib import Path


class Config:
    data_root = Path("./data/Synapse")
    data_path = data_root / "train_npz"
    list_dir = data_root / "lists/lists_Synapse"
    volume_path = data_root / "test_vol_h5"
    pretrained_path = Path("./pre_trained_weights/vmamba_small_e238_ema.pth")
    output_dir = Path("./results")

    input_size = 224
    input_channels = 1
    num_classes = 9
    z_spacing = 1.0
    model = {
        "in_chans": input_channels,
        "num_classes": num_classes,
        "depths": [2, 4, 8, 2],
        "depths_decoder": [2, 4, 4, 2],
        "dims": [96, 192, 384, 768],
        "d_state": 16,
        "drop_path_rate": 0.2,
    }

    seed = 42
    epochs = 300
    batch_size = 32
    num_workers = 8
    learning_rate = 1e-3
    weight_decay = 1e-2
    scheduler_t_max = 100
    scheduler_eta_min = 1e-5
    validation_interval = 100


config = Config()
