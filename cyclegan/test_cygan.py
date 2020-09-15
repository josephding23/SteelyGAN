import torch
from util.data.dataset import SteelyDataset, get_dataset
import shutil
from util.analysis.tonality import evaluate_tonal_scale_of_data
from util.toolkits.data_convert import save_midis
from util.toolkits.database import get_md5_of
from cyclegan.cygan_model import CycleGAN, CyganConfig
from classify.classify_model import Classify
from networks.SteelyGAN import Discriminator, Generator
import matplotlib.pyplot as plt
import numpy as np
from util.analysis.tonality import *
from torch.optim import lr_scheduler, Adam
from util.toolkits.data_convert import *


def test_sample_song_old():
    dataset = SteelyDataset('rock', 'jazz', 'test', False)

    cyclegan = CycleGAN()
    cyclegan.continue_from_latest_checkpoint()

    converted_dir = '../data/converted_midi'

    for index in range(10):
        data = dataset[index + 2000]
        dataA, dataB = data[0, :, :], data[1, :, :]
        # print(torch.unsqueeze(torch.from_numpy(dataA), 0).shape)
        dataA2B = cyclegan.generator_A2B(
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(dataA), 0), 0).to(device='cuda',
                                                                               dtype=torch.float)).cpu().detach().numpy()[
                  0, 0, :, :]
        dataB2A = cyclegan.generator_B2A(
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(dataB), 0), 0).to(device='cuda',
                                                                               dtype=torch.float)).cpu().detach().numpy()[
                  0, 0, :, :]
        midi_A_path = converted_dir + '/midi_A_' + str(index) + '.mid'
        midi_A2B_path = converted_dir + '/midi_A2B_' + str(index) + '.mid'

        midi_B_path = converted_dir + '/midi_B_' + str(index) + '.mid'
        midi_B2A_path = converted_dir + '/midi_B2A_' + str(index) + '.mid'

        tonality_A = evaluate_tonal_scale_of_data(dataA)
        tonality_A2B = evaluate_tonal_scale_of_data(dataA2B)

        tonality_B = evaluate_tonal_scale_of_data(dataB)
        tonality_B2A = evaluate_tonal_scale_of_data(dataB2A)

        print(tonality_A, tonality_A2B)
        print(tonality_B, tonality_B2A)

        # plot_data(dataA)
        # plot_data(dataA2B)

        generate_midi_segment_from_tensor(dataA, midi_A_path)
        generate_midi_segment_from_tensor(dataA2B, midi_A2B_path)
        generate_midi_segment_from_tensor(dataB, midi_B_path)
        generate_midi_segment_from_tensor(dataB2A, midi_B2A_path)


def test_whole_song(model_name, ):
    test_dict = [
        {
            'performer': 'Sex Pistols',
            'song': "Anarchy In The Uk",
            'genre': 'punk',
            # 'path': '../data/original_midi/Basket Case - Green Day.mid',
            'path': '../data/original_midi/Anarchy In The Uk - Sex Pistols.mid',
            'direction': 'AtoB',
            'group': 2
        },
        {
            'performer': 'Beethoven',
            'song': "Symphony No.6 in F 'Pastorale', Op.68 --1.Allegro ma non troppo",
            'genre': 'classical',
            'path':
                "../data/original_midi/Symphony No.6 in F 'Pastorale', Op.68 --1.Allegro ma non troppo - Beethoven.mid",
            'direction': 'BtoA',
            'group': 2
        },
        {
            'performer': 'AC DC',
            'song': 'Back In Black',
            'path': '../data/original_midi/Back In Black - AC DC.mid',
            'direction': 'AtoB',
            'group': 3
        },
        {
            'performer': 'Frank Sinatra',
            'song': 'Fly Me To The Moon',
            'genre': 'jazz',
            'path': '../data/original_midi/Fly Me To The Moon - Frank Sinatra.mid',
            'direction': 'BtoA',
            'group': 3
        }
    ]

    for test_info in test_dict:
        original_path = test_info['path']

        transformed_path = '../data/converted_midi/' + f"{model_name} - {test_info['song']} - {test_info['performer']}.mid"

        ori_data = generate_data_from_midi(original_path)
        ori_data = np.expand_dims(ori_data, 1)

        continue_train = True
        device = 'GPU'
        opt = CyganConfig(model_name, test_info['group'], continue_train)
        cyclegan = CycleGAN(opt, device)
        cyclegan.continue_from_latest_checkpoint()

        direction = test_info['direction']

        cyclegan.generator_A2B.eval()
        cyclegan.generator_B2A.eval()
        if direction == 'AtoB':
            transformed_data = cyclegan.generator_A2B(torch.from_numpy(ori_data.copy()).to(
                    device='cuda',  dtype=torch.float)).cpu().detach().numpy()
        else:
            transformed_data = cyclegan.generator_B2A(torch.from_numpy(ori_data.copy()).to(
                    device='cuda', dtype=torch.float)).cpu().detach().numpy()

        save_midis(transformed_data, transformed_path)
        save_midis(ori_data, original_path)


def test_lr():
    model = Discriminator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=4e-08)
    lr_list = []
    for epoch in range(20):
        scheduler.step(epoch)
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    plt.plot(range(20), lr_list)
    plt.show()


if __name__ == '__main__':
    for model_name in ['steely_gan', 'SMGT']:
        test_whole_song(model_name)
