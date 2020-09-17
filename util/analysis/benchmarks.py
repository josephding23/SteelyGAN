from steely_util.toolkits.data_convert import *


def ratio_of_empty_bars(path):
    bars_info_matrix = get_bar_info_from_midi(path)
    bars_num = bars_info_matrix.shape[0]
    empty_bars_num = 0
    for bar in range(bars_num):
        current_bar_info = bars_info_matrix[bar, :, :]
        if np.any(current_bar_info) == 0:
            empty_bars_num += 1
    return empty_bars_num / bars_num


def number_of_used_pitch_classses_per_bar(path):
    bars_info_matrix = get_bar_info_from_midi(path)
    bars_num = bars_info_matrix.shape[0]
    bars_pitch_classes_info = np.zeros(shape=(bars_num, 12))
    bars_pitches_num = []
    for bar in range(bars_num):
        bar_info = bars_info_matrix[bar, :, :]
        for time in range(16):
            for pitch in range(84):
                if bar_info[time, pitch] == 1 and bars_pitch_classes_info[bar, pitch % 12] == 0:
                    bars_pitch_classes_info[bar, pitch % 12] = 1
                else:
                    pass
    for bar in range(bars_num):
        current_bar_pitches = bars_pitch_classes_info[bar, :]
        bars_pitches_num.append(len(current_bar_pitches.nonzero()[0]))

    return np.mean(bars_pitches_num)


def in_scale_notes_ratio(path):
    bars_info_matrix = get_bar_info_from_midi(path)
    bars_num = bars_info_matrix.shape[0]
    all_notes_num = 0
    in_scale_notes_num = 0

    for bar in range(bars_num):
        bar_info = bars_info_matrix[bar, :, :]
        for time in range(16):
            for pitch in range(84):
                if bar_info[time, pitch] != 0:
                    all_notes_num += 1
                    if pitch % 12 in [0, 2, 4, 5, 7, 9, 11]:
                        in_scale_notes_num += 1
    return in_scale_notes_num / all_notes_num


def tonal_distance():
    pass


def test_empty_bars():
    path = '../../data/converted_midi/steely_gan - Anarchy In The Uk - Sex Pistols.mid'
    print(ratio_of_empty_bars(path), in_scale_notes_ratio(path), number_of_used_pitch_classses_per_bar(path))


if __name__ == '__main__':
    test_empty_bars()