from prepare_datasets import *
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from profiling_and_attack import *

tf.keras.backend.set_floatx('float64')


class TrainCGAN:

    def __init__(self, args, models, dir_results):
        self.args = args
        self.datasets = PrepareDatasets(self.args)
        self.models = models
        self.dir_results = dir_results

        """ Metrics to assess quality of profiling attack: Max SNR, Guessing entropy, Ntraces_GE = 1, Perceived Information """
        self.snr_features_share_1 = []
        self.snr_features_share_2 = []
        self.max_snr_share_1 = []
        self.max_snr_share_2 = []

        self.ge_fake = []
        self.nt_fake = []
        self.pi_fake = []

        self.ge_real = []
        self.nt_real = []
        self.pi_real = []

        self.ge_real_original = []
        self.nt_real_original = []
        self.pi_real_original = []

        self.ge_real_ta = []
        self.nt_real_ta = []
        self.pi_real_ta = []

        """ Just for plot """
        self.x_axis_epochs = []

        """ Accuracy for real and synthetic data """
        self.real_acc = []
        self.fake_acc = []

        """ Generator and Discriminator Losses """
        self.g_loss = []
        self.d_loss = []

    def generate_reference_samples(self, batch_size):
        rnd = np.random.randint(0, self.datasets.dataset_reference.n_profiling - batch_size)
        features = self.datasets.features_reference_profiling[rnd:rnd + batch_size]
        labels = self.datasets.dataset_reference.profiling_labels[rnd:rnd + batch_size]
        return [features, labels]

    def generate_target_samples(self, batch_size):
        rnd = np.random.randint(0, self.datasets.dataset_target.n_profiling - batch_size)
        traces = self.datasets.dataset_target.x_profiling[rnd:rnd + batch_size]
        labels = self.datasets.dataset_target.profiling_labels[rnd:rnd + batch_size]
        return [traces, labels]

    @tf.function
    def train_step(self, traces_batch, label_traces, features, label_features):

        with tf.GradientTape() as disc_tape:
            fake_features = self.models.generator(traces_batch)
            real_output = self.models.discriminator([label_features, features])
            fake_output = self.models.discriminator([label_traces, fake_features])
            disc_loss = self.models.discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.models.discriminator.trainable_variables)
        self.models.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.models.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            fake_features = self.models.generator(traces_batch)
            fake_output = self.models.discriminator([label_traces, fake_features])
            gen_loss = self.models.generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.models.generator.trainable_variables)
        self.models.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.models.generator.trainable_variables))

        self.models.fake_accuracy_metric.update_state(tf.zeros_like(fake_features), fake_output)
        self.models.real_accuracy_metric.update_state(tf.ones_like(features), real_output)

        return gen_loss, disc_loss

    def compute_snr_reference_features(self):
        batch_size_reference = 10000

        # prepare traces from target dataset
        rnd_reference = random.randint(0, len(self.datasets.dataset_reference.x_profiling) - batch_size_reference)
        features_reference = self.datasets.features_reference_profiling[rnd_reference:rnd_reference + batch_size_reference]

        snr_reference_features_share_1 = snr_fast(features_reference,
                                                  self.datasets.dataset_reference.share1_profiling[self.datasets.target_byte_reference,
                                                  rnd_reference:rnd_reference + batch_size_reference])
        snr_reference_features_share_2 = snr_fast(features_reference,
                                                  self.datasets.dataset_reference.share2_profiling[self.datasets.target_byte_reference,
                                                  rnd_reference:rnd_reference + batch_size_reference])
        plt.plot(snr_reference_features_share_1)
        plt.plot(snr_reference_features_share_2)

        plt.xlim([1, self.datasets.features_dim])
        plt.savefig(f"{self.dir_results}/snr_reference_features.png")
        plt.close()

        np.savez(f"{self.dir_results}/snr_reference_features.npz", snr_reference_features_share_1=snr_reference_features_share_1,
                 snr_reference_features_share_2=snr_reference_features_share_2)

    def compute_snr_target_features(self, epoch, synthetic_traces=True):
        batch_size_target = 10000

        # prepare traces from target dataset
        rnd_target = random.randint(0, len(self.datasets.dataset_target.x_profiling) - batch_size_target)

        if synthetic_traces:
            traces_target = self.datasets.dataset_target.x_profiling[rnd_target:rnd_target + batch_size_target]
            features_target = self.models.generator.predict([traces_target])
        else:
            features_target = self.datasets.features_target_profiling[rnd_target:rnd_target + batch_size_target]

        snr_target_features_share_1 = snr_fast(features_target,
                                               self.datasets.dataset_target.share1_profiling[self.datasets.target_byte_target,
                                               rnd_target:rnd_target + batch_size_target]).tolist()
        snr_target_features_share_2 = snr_fast(features_target,
                                               self.datasets.dataset_target.share2_profiling[self.datasets.target_byte_target,
                                               rnd_target:rnd_target + batch_size_target]).tolist()

        plt.plot(snr_target_features_share_1)
        plt.plot(snr_target_features_share_2)
        plt.xlim([1, self.datasets.features_dim])
        if synthetic_traces:
            plt.savefig(f"{self.dir_results}/snr_target_features_fake_{epoch}.png")
            plt.close()
            self.snr_features_share_1.append(snr_target_features_share_1)
            self.snr_features_share_2.append(snr_target_features_share_2)
            np.savez(f"{self.dir_results}/snr_target_features_fake.npz", snr_target_features_share_1=self.snr_features_share_1,
                     snr_target_features_share_2=self.snr_features_share_2)
        else:
            plt.savefig(f"{self.dir_results}/snr_target_features_real_{epoch}.png")
            plt.close()

            np.savez(f"{self.dir_results}/snr_target_features_real.npz", snr_target_features_share_1=snr_target_features_share_1,
                     snr_target_features_share_2=snr_target_features_share_2)

        if synthetic_traces:
            self.max_snr_share_1.append(np.max(snr_target_features_share_1))
            self.max_snr_share_2.append(np.max(snr_target_features_share_2))
            plt.plot(self.max_snr_share_1, label="Max SNR Share 1")
            plt.plot(self.max_snr_share_2, label="Max SNR Share 2")
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("SNR")
            plt.savefig(f"{self.dir_results}/max_snr_shares.png")
            plt.close()
            np.savez(f"{self.dir_results}/max_snr_shares.npz", max_snr_share_1=self.max_snr_share_1,
                     max_snr_share_2=self.max_snr_share_2)

    def attack_eval(self, epoch):

        ge_fake, nt_fake, pi_fake, ge_vector_fake = attack(self.datasets, self.models.generator, self.datasets.features_dim)
        self.ge_fake.append(ge_fake)
        self.nt_fake.append(nt_fake)
        self.pi_fake.append(pi_fake)

        self.x_axis_epochs.append(epoch + 1)

        plt.plot(self.x_axis_epochs, self.ge_fake, label="CGAN-SCA")
        plt.legend()
        plt.xlabel("CGAN Training Epoch")
        plt.ylabel("Guessing Entropy")
        plt.savefig(f"{self.dir_results}/ge.png")
        plt.close()

        plt.plot(self.x_axis_epochs, self.nt_fake, label="CGAN-SCA")
        plt.legend()
        plt.xlabel("CGAN Training Epoch")
        plt.ylabel("Number of Traces for GE=1")
        plt.yscale('log')
        plt.savefig(f"{self.dir_results}/nt.png")
        plt.close()

        plt.plot(self.x_axis_epochs, self.pi_fake, label="CGAN-SCA")
        plt.legend()
        plt.xlabel("CGAN Training Epoch")
        plt.ylabel("Perceived Information")
        plt.savefig(f"{self.dir_results}/pi.png")
        plt.close()

        np.savez(f"{self.dir_results}/metrics.npz",
                 ge_fake=self.ge_fake,
                 nt_fake=self.nt_fake,
                 pi_fake=self.pi_fake
                 )

        plt.plot(ge_vector_fake, label="CGAN-SCA")
        plt.legend()
        plt.xscale('log')
        plt.xlabel("Attack Traces")
        plt.ylabel("Guessing Entropy")
        plt.savefig(f"{self.dir_results}/ge_epoch_{epoch}.png")
        plt.close()

        np.savez(f"{self.dir_results}/ge_vector_epoch_{epoch}.npz", ge_vector_fake=ge_vector_fake)

        # if self.args["random_search_generator"] or self.args["random_search_discriminator"]:
        #     ge_fake, nt_fake, pi_fake, ge_vector_fake = attack(self.datasets, self.models.generator, self.datasets.features_dim)
        #     self.ge_fake.append(ge_fake)
        #     self.nt_fake.append(nt_fake)
        #     self.pi_fake.append(pi_fake)
        #
        #     self.x_axis_epochs.append(epoch + 1)
        #
        #     plt.plot(self.x_axis_epochs, self.ge_fake, label="CGAN-SCA")
        #     plt.legend()
        #     plt.xlabel("CGAN Training Epoch")
        #     plt.ylabel("Guessing Entropy")
        #     plt.savefig(f"{self.dir_results}/ge.png")
        #     plt.close()
        #
        #     plt.plot(self.x_axis_epochs, self.nt_fake, label="CGAN-SCA")
        #     plt.legend()
        #     plt.xlabel("CGAN Training Epoch")
        #     plt.ylabel("Number of Traces for GE=1")
        #     plt.yscale('log')
        #     plt.savefig(f"{self.dir_results}/nt.png")
        #     plt.close()
        #
        #     plt.plot(self.x_axis_epochs, self.pi_fake, label="CGAN-SCA")
        #     plt.legend()
        #     plt.xlabel("CGAN Training Epoch")
        #     plt.ylabel("Perceived Information")
        #     plt.savefig(f"{self.dir_results}/pi.png")
        #     plt.close()
        #
        #     np.savez(f"{self.dir_results}/metrics.npz",
        #              ge_fake=self.ge_fake,
        #              nt_fake=self.nt_fake,
        #              pi_fake=self.pi_fake
        #              )
        #
        #     plt.plot(ge_vector_fake, label="CGAN-SCA")
        #     plt.legend()
        #     plt.xscale('log')
        #     plt.xlabel("Attack Traces")
        #     plt.ylabel("Guessing Entropy")
        #     plt.savefig(f"{self.dir_results}/ge_epoch_{epoch}.png")
        #     plt.close()
        #
        #     np.savez(f"{self.dir_results}/ge_vector_epoch_{epoch}.npz", ge_vector_fake=ge_vector_fake)
        # else:
        #
        #     ge_real_ta, nt_real_ta, pi_real_ta, ge_vector_real_ta = template_attack(self.datasets)
        #     ge_fake, nt_fake, pi_fake, ge_vector_fake = attack(self.datasets, self.models.generator, self.datasets.features_dim)
        #     ge_real, nt_real, pi_real, ge_vector_real = attack(self.datasets, self.models.generator, self.datasets.features_dim,
        #                                                        synthetic_traces=False)
        #     ge_real_original, nt_real_original, pi_real_original, ge_vector_real_original = attack(self.datasets, self.models.generator,
        #                                                                                            self.datasets.dataset_target.ns,
        #                                                                                            synthetic_traces=False,
        #                                                                                            original_traces=True)
        #     self.ge_fake.append(ge_fake)
        #     self.nt_fake.append(nt_fake)
        #     self.pi_fake.append(pi_fake)
        #     self.ge_real.append(ge_real)
        #     self.nt_real.append(nt_real)
        #     self.pi_real.append(pi_real)
        #     self.ge_real_original.append(ge_real_original)
        #     self.nt_real_original.append(nt_real_original)
        #     self.pi_real_original.append(pi_real_original)
        #     self.ge_real_ta.append(ge_real_ta)
        #     self.nt_real_ta.append(nt_real_ta)
        #     self.pi_real_ta.append(pi_real_ta)
        #
        #     self.x_axis_epochs.append(epoch + 1)
        #
        #     plt.plot(self.x_axis_epochs, self.ge_fake, label="fake")
        #     plt.plot(self.x_axis_epochs, self.ge_real, label="real")
        #     plt.plot(self.x_axis_epochs, self.ge_real_original, label="real raw")
        #     plt.plot(self.x_axis_epochs, self.ge_real_ta, label="real TA (LDA)")
        #     plt.legend()
        #     plt.xlabel("CGAN Training Epoch")
        #     plt.ylabel("Guessing Entropy")
        #     plt.savefig(f"{self.dir_results}/ge.png")
        #     plt.close()
        #
        #     plt.plot(self.x_axis_epochs, self.nt_fake, label="fake")
        #     plt.plot(self.x_axis_epochs, self.nt_real, label="real")
        #     plt.plot(self.x_axis_epochs, self.nt_real_original, label="real raw")
        #     plt.plot(self.x_axis_epochs, self.nt_real_ta, label="TA (LDA)")
        #     plt.legend()
        #     plt.xlabel("CGAN Training Epoch")
        #     plt.ylabel("Number of Traces for GE=1")
        #     plt.yscale('log')
        #     plt.savefig(f"{self.dir_results}/nt.png")
        #     plt.close()
        #
        #     plt.plot(self.x_axis_epochs, self.pi_fake, label="fake")
        #     plt.plot(self.x_axis_epochs, self.pi_real, label="real")
        #     plt.plot(self.x_axis_epochs, self.pi_real_original, label="real raw")
        #     plt.plot(self.x_axis_epochs, self.pi_real_ta, label="TA (LDA)")
        #     plt.legend()
        #     plt.xlabel("CGAN Training Epoch")
        #     plt.ylabel("Perceived Information")
        #     plt.savefig(f"{self.dir_results}/pi.png")
        #     plt.close()
        #
        #     np.savez(f"{self.dir_results}/metrics.npz",
        #              ge_fake=self.ge_fake,
        #              nt_fake=self.nt_fake,
        #              pi_fake=self.pi_fake,
        #              ge_real=self.ge_real,
        #              nt_real=self.nt_real,
        #              pi_real=self.pi_real,
        #              ge_real_original=self.ge_real_original,
        #              nt_real_original=self.nt_real_original,
        #              pi_real_original=self.pi_real_original,
        #              ge_real_ta=self.ge_real_ta,
        #              nt_real_ta=self.nt_real_ta,
        #              pi_real_ta=self.pi_real_ta
        #              )
        #
        #     plt.plot(ge_vector_fake, label="fake")
        #     plt.plot(ge_vector_real, label="real")
        #     plt.plot(ge_vector_real_original, label="real raw")
        #     plt.plot(ge_vector_real_ta, label="real TA (LDA)")
        #     plt.legend()
        #     plt.xscale('log')
        #     plt.xlabel("Attack Traces")
        #     plt.ylabel("Guessing Entropy")
        #     plt.savefig(f"{self.dir_results}/ge_epoch_{epoch}.png")
        #     plt.close()
        #
        #     np.savez(f"{self.dir_results}/ge_vector_epoch_{epoch}.npz",
        #              ge_vector_fake=ge_vector_fake,
        #              ge_vector_real=ge_vector_real,
        #              ge_vector_real_original=ge_vector_real_original,
        #              ge_vector_real_ta=ge_vector_real_ta
        #              )

    def train(self):
        training_set_size = max(self.datasets.dataset_reference.n_profiling, self.datasets.dataset_target.n_profiling)

        # determine half the size of one batch, for updating the discriminator
        batch_size = self.args["batch_size"]
        n_batches = int(training_set_size / batch_size)

        epoch_snr_step = 10 if self.args["random_search_generator"] or self.args["random_search_discriminator"] else 1

        # manually enumerate epochs
        for e in range(self.args["epochs"]):
            for b in range(n_batches):
                [features_reference, labels_reference] = self.generate_reference_samples(batch_size)
                [traces_target, labels_target] = self.generate_target_samples(batch_size)

                # Custom training step for speed and versatility
                g_loss, d_loss = self.train_step(traces_target, labels_target, features_reference, labels_reference)

                if (b + 1) % 100 == 0:
                    self.real_acc.append(self.models.real_accuracy_metric.result())
                    self.fake_acc.append(self.models.fake_accuracy_metric.result())
                    self.g_loss.append(g_loss)
                    self.d_loss.append(d_loss)

                    plt.plot(self.real_acc, label="Real")
                    plt.plot(self.fake_acc, label="Fake")
                    plt.axhline(y=0.5, linestyle="dashed", color="black")
                    plt.legend()
                    plt.savefig(f"{self.dir_results}/acc.png")
                    plt.close()

                    plt.plot(self.g_loss, label="g_loss")
                    plt.plot(self.d_loss, label="d_Loss")
                    plt.legend()
                    plt.savefig(f"{self.dir_results}/loss.png")
                    plt.close()

                    print(
                        f"epoch: {e}, batch: {b}, d_loss: {d_loss}, g_loss: {g_loss}, real_acc: {self.models.real_accuracy_metric.result()}, fake_acc: {self.models.fake_accuracy_metric.result()}")
                    np.savez(f"{self.dir_results}/acc_and_loss.npz",
                             g_loss=self.g_loss, d_loss=self.d_loss,
                             real_acc=self.real_acc, fake_acc=self.fake_acc)

            # Split eval steps up as attacking takes significant time while snr computation is fast
            if e == 0:
                self.compute_snr_reference_features()
                self.compute_snr_target_features(e, synthetic_traces=False)
            if (e + 1) % epoch_snr_step == 0:
                self.compute_snr_target_features(e)
                # self.attack_eval_synthetic(e)
            if (e + 1) % 50 == 0:
                self.attack_eval(e)
                if not self.args["random_search_generator"] and not self.args["random_search_discriminator"]:
                    self.models.generator.save(
                        f"{self.dir_results}/generator_{self.datasets.traces_target_dim}_{self.datasets.traces_reference_dim}_epoch_{e}.h5")
        if not self.args["random_search_generator"] and not self.args["random_search_discriminator"]:
            self.models.generator.save(
                f"{self.dir_results}/generator_{self.datasets.traces_target_dim}_{self.datasets.traces_reference_dim}_epoch_{self.args['epochs'] - 1}.h5")
