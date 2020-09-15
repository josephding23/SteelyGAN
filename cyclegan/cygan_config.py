import os


class CyganConfig(object):
    def __init__(self, name, genre_group, continue_train):

        ##########################
        # Info

        self.name = name
        assert self.name in ['steely_gan', 'SMGT']

        self.dataset_name = 'free_midi_library'

        self.genre_group = genre_group

        self.continue_train = continue_train

        if self.genre_group == 1:
            self.genreA = 'metal'
            self.genreB = 'country'

        elif self.genre_group == 2:
            self.genreA = 'punk'
            self.genreB = 'classical'

        else:
            self.genreA = 'rock'
            self.genreB = 'jazz'

        self.dataset_mode = 'unaligned'
        self.track_merged = True

        self.time_step = 120
        self.bar_length = 4
        self.note_valid_range = (24, 108)
        self.note_valid_length = 84
        self.instr_num = 5

        self.phase = 'train'

        self.direction = 'AtoB'



        ###########################

        ###########################
        # Structure

        self.model = 'base'  # three different models, base, partial, full

        self.use_image_pool = True
        self.image_pool_info = 'pooled' if self.use_image_pool else 'not_pooled'
        self.image_pool_max_size = 20

        self.bat_unit_eta = 0.2

        ##########################

        ##########################
        # Train

        self.gaussian_std = 1

        self.sigma_c = 1.0
        self.sigma_d = 1.0

        self.gpu = True

        self.beta1 = 0.5                     # Adam optimizer beta1 & 2
        self.beta2 = 0.999

        self.lr = 0.0001
        self.milestones = [2, 5, 8, 11, 13, 15, 17, 19, 20]
        self.gamma = 0.5

        self.weight_decay = 0.0

        self.no_flip = True
        self.num_threads = 0
        self.batch_size = 8
        self.max_epoch = 40
        self.epoch_step = 5

        self.data_shape = (self.batch_size, 1, 64, 84)
        self.input_shape = (1, 64, 84)

        self.plot_every = 500                # iterations
        self.save_every = 1                  # epochs

        self.start_epoch = 0

        ##########################

        ##########################
        # Save Paths

        self.save_path = '../checkpoints/'

        self.log_path = self.save_path + '/info.log'
        self.loss_save_path = self.save_path + '/losses.json'

        self.test_path = self.save_path + '/test_results'
        self.test_save_path = self.test_path + '/' + self.direction

        ##########################


if __name__ == '__main__':
    config = Config()
    config.genreA = 'rock'
    config.genreB = 'jazz'

