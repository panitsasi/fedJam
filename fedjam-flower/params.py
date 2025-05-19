# params.py

param_grid = [
    {
        #### MOBILENETV3 SMALL 100 ####
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "timm/mobilenetv3_small_100.lamb_in1k",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 10,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): True,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 1,


        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 10,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },

    {
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "timm/mobilenetv3_small_100.lamb_in1k",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 20,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): True,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 1,

        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 20,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },


    {
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "timm/mobilenetv3_small_100.lamb_in1k",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 50,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): True,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 1,

        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 50,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },
    
    ### VIT-TINY 100 ####

    {
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "WinKawaks/vit-tiny-patch16-224",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 10,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): False,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 1,

        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 10,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },

    {
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "WinKawaks/vit-tiny-patch16-224",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 20,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): False,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 1,

        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 20,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },

    {
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "WinKawaks/vit-tiny-patch16-224",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 50,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): False,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 1,

        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 50,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },

{
        #### MOBILENETV3 SMALL 100 ####
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "timm/mobilenetv3_small_100.lamb_in1k",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 10,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): True,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 2,


        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 10,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },

    {
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "timm/mobilenetv3_small_100.lamb_in1k",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 20,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): True,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 2,

        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 20,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },


    {
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "timm/mobilenetv3_small_100.lamb_in1k",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 50,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): True,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 2,

        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 50,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },
    
    ### VIT-TINY 100 ####

    {
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "WinKawaks/vit-tiny-patch16-224",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 10,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): False,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 2,

        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 10,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },

    {
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "WinKawaks/vit-tiny-patch16-224",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 20,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): False,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 2,

        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 20,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },

    {
        # tool.flwr.app.config
        ("app.config", "num-server-rounds"): 50,
        ("app.config", "fraction-fit"): 1,
        ("app.config", "local-epochs"): 1,
        ("app.config", "model_name"): "WinKawaks/vit-tiny-patch16-224",
        ("app.config", "learning-rate-max"): 5e-4,
        ("app.config", "learning-rate-min"): 1e-6,
        ("app.config", "batch_size"): 128,
        ("app.config", "dataset_version"): "dataset_v3",
        ("app.config", "data_dir"): "/home/ioannis/Desktop/spectrograms/dataset_v3/spectro_flower_all",
        ("app.config", "fft"): 0,
        ("app.config", "gain"): 0,
        ("app.config", "num_clients"): 50,
        ("app.config", "is_lora"): False,
        ("app.config", "is_timm"): False,
        ("app.config", "is_warmup"): False,
        ("app.config", "random_seed"): 43,
        ("app.config", "classes_per_partition"): 2,

        # tool.flwr.federations.local-simulation
        ("federations.local-simulation.options", "num-supernodes"): 50,
        ("federations.local-simulation.options.backend.client-resources", "num-cpus"): 7,
        ("federations.local-simulation.options.backend.client-resources", "num-gpus"): 0.3,
    },

]