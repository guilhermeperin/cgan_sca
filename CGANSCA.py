from train_cgan import *
from create_models import *


class CGANSCA:

    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.main_path = self.args["results_root_path"]

    def train_cgan(self):
        dir_results = create_directory_results(self.args, self.main_path)
        models = CreateModels(self.args, dir_results)
        np.savez(f"{dir_results}/args.npz", args=self.args)
        train_cgan = TrainCGAN(self.args, models, dir_results)
        train_cgan.train()
