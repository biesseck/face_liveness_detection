from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)

# config.network = "r50"   # original
config.network = "r18"     # Bernardo

config.resume = False
config.output = None

# config.embedding_size = 512  # original
config.embedding_size = 256    # Bernardo

config.sample_rate = 1.0
# config.fp16 = True       # original
config.fp16 = False        # Bernardo
config.momentum = 0.9
config.weight_decay = 5e-4

# config.batch_size = 128  # original
# config.batch_size = 64   # Bernardo
# config.batch_size = 32   # Bernardo
# config.batch_size = 16   # Bernardo
# config.batch_size = 8   # Bernardo
config.batch_size = 4   # Bernardo

config.lr = 0.1

# config.verbose = 2000  # original for 5.1M images
# config.verbose = 100     # Bernardo
config.verbose = 1    # Bernardo

config.dali = False
config.dali_aug = False

# config.rec = "/train_tmp/ms1m-retinaface-t1"                                                # original
config.train_dataset = 'oulu-npu_frames_mini'                                                 # Bernardo
config.protocol_id = 1                                                                        # Bernardo
config.dataset_path = '/datasets_ufpr/liveness/oulu-npu_frames_crop224x224'  # Bernardo
# config.frames_path = ''  # Bernardo

# config.img_size = 112        # Bernardo
config.img_size = 224          # Bernardo

# config.num_classes = 93431   # original
config.num_classes = 2         # (live or spoof) Bernardo

# config.num_image = 5179510   # original
config.num_image = 1800        # Bernardo

# config.max_epoch = 20
# config.max_epoch = 50
# config.max_epoch = 100
config.max_epoch = 300

config.warmup_epoch = 0
# config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]



# WandB Logger
config.wandb_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
config.suffix_run_name = None
config.using_wandb = False
config.wandb_entity = "entity"
config.wandb_project = "project"
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False # resume wandb run: Only if the you wand t resume the last run that it was interrupted