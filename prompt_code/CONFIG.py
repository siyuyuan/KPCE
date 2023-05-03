

class CONFIG:
    def __init__(self):
        self.log_path = "log.txt"
        self.tokenizer_path = "bert-base-chinese"
        self.model_config_path = "config/model_config.json"
        self.device = "gpu"
        self.saved_model_path = "output_enhance"
        self.pretrained_model_path = "bert-base-chinese"
        self.test = True
        self.num_workers = 0
        self.train_data_path = "dataset/train_mrc_topic.txt"
        self.dev_data_path = "dataset/dev_mrc_topic.txt"
        self.test_data_path = "dataset/test_mrc_topic.txt"
        self.test_size = 0.1
        self.batch_size=16
        self.epochs = 5
        self.save_model_dic="output_topic_vector/model_epoch5/Prompt_MRCModel.pth"
        self.lr=1.5e-4
        self.warmup_steps=2000
        self.log_step=1
        self.max_grad_norm=1.0
        self.gradient_accumulation=1
        self.topk=8
        self.test_max_len=20
        self.save_mode=True
        self.n_ctx=256
        self.num_classification=19
        self.together=False
