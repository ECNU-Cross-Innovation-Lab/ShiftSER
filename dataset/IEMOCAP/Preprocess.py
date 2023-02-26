import os
import re
import utils
import argparse
import torchaudio
import numpy as np
import pandas as pd
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='path of IEMOCAP dataset')
parser.add_argument('--second', type=float, default=7.5,help="length of the sample (in second)")


def Raw2Df(indir):
    start_times, end_times, wav_file_names, emotions = [], [], [], []
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
    for sess in range(1, 6):
        emo_evaluation_dir = os.path.join(indir,'Session{}/dialog/EmoEvaluation/'.format(sess))
        emo_sentences_dir = os.path.join(indir,'Session{}/sentences/wav/'.format(sess))
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        for file in evaluation_files:
            with open(emo_evaluation_dir + file) as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            emo_wav_dir = emo_sentences_dir + file.split('.')[0] + '/'
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                wav_file_name = emo_wav_dir + wav_file_name + '.wav'
                start_time, end_time = start_end_time[1:-1].split('-')
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)

    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion'])

    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['wav_file'] = wav_file_names
    df_iemocap['emotion'] = emotions
    return df_iemocap


def Df2Csv(df_iemocap, csvdir):
    df = df_iemocap.copy()
    emotion_id = {'hap': 0, 'ang': 1, 'sad': 2, 'neu': 3}
    df = df[(df.emotion == 'ang') | (df.emotion == 'sad') | (df.emotion == 'exc') | (df.emotion == 'hap') |
            (df.emotion == 'neu')]
    df.loc[df['emotion'] == 'exc', 'emotion'] = 'hap' # merge
    df['emotion'] = df['emotion'].map(emotion_id)
    df.to_csv(csvdir, index=False, encoding="utf_8_sig")


def probe_lenghth(csvdir):
    lengths = []
    df = pd.read_csv(csvdir, usecols=['wav_file', 'emotion'])
    for i in range(1, 6):
        sess = 'Session{}'.format(i)
        df_temp = df[df['wav_file'].str.contains(sess)]
        L = df_temp.values.T.tolist()
        audio_links = L[0]
        for link in audio_links:
            wav, sample_rate = torchaudio.load(link)
            lengths.append(wav.shape[1] / sample_rate)
    print(np.mean(lengths))  # 4.5s
    print(np.std(lengths))  # 3.2s


def Csv2Pickle(csvdir, pikdir, second):
    SessionMap = {}
    df = pd.read_csv(csvdir, usecols=['wav_file', 'emotion'])

    for i in range(1, 6):
        sess = 'Session{}'.format(i)
        DataMap = {}
        df_temp = df[df['wav_file'].str.contains(sess)]
        L = df_temp.values.T.tolist()
        audio_links = L[0]
        DataMap['data'] = []
        DataMap['length'] = []
        for link in audio_links:
            feature, length = utils.load_wav(link, second)
            DataMap['data'].append(feature)
            DataMap['length'].append(length)

        DataMap['labels_emotion'] = L[1]
        SessionMap[sess] = DataMap
        print('Session{} Completed!'.format(i))
    with open(pikdir, 'wb') as f:
        pkl.dump(SessionMap, f)


if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.path
    second = args.second
    csvdir = './IEMOCAP.csv'
    pikdir = './wavfeature' + '_' + str(second) + '.pkl'
    print(f'output pickle: {pikdir}')
    # probe_lenghth(csvdir)
    df_iemocap = Raw2Df(indir)
    Df2Csv(df_iemocap, csvdir)
    Csv2Pickle(csvdir, pikdir, second)
