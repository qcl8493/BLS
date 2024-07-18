# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gevent
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
# from pyst import st
import warnings
import pandas as pd
import shutil
import tool_calculate
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns

# matplotlib.pyplot
# 1. 计算时域特征
def get_time_domain_features(data_re, data_im):
    # 1.1 均值mean
    mean_re_td = np.mean(data_re)
    mean_im_td = np.mean(data_im)

    # 1.2 中位数mid
    mid_re_td = np.median(data_re)
    mid_im_td = np.median(data_im)

    # 1.3 方差var
    var_re_td = np.var(data_re)
    var_im_td = np.var(data_im)

    # 1.4 均方根rms
    rms_re_td = np.sqrt(np.mean(np.square(data_re)))
    rms_im_td = np.sqrt(np.mean(np.square(data_im)))

    # 1.5 标准差std
    std_re_td = np.std(data_re)
    std_im_td = np.std(data_im)

    # 1.6 波形因子waveform
    waveform_re_td = rms_re_td / mean_re_td
    waveform_im_td = rms_im_td / mean_im_td

    # 1.7 峭度因子kur
    kur_re_td = scipy.stats.kurtosis(data_re)
    kur_im_td = scipy.stats.kurtosis(data_im)

    # 1.8 偏度因子ske
    ske_re_td = scipy.stats.skew(data_re)
    ske_im_td = scipy.stats.skew(data_im)

    return (mean_re_td, mean_im_td, mid_re_td, mid_im_td,
            var_re_td, var_im_td, rms_re_td, rms_im_td,
            std_re_td, std_im_td, waveform_re_td, waveform_im_td,
            kur_re_td, kur_im_td, ske_re_td, ske_im_td)

# 2. 计算频域特征
def get_frequncy_domain_feature(data_re, data_im, N, fs):
    signal = data_re + data_im * 1j
    fft_res = np.fft.fft(signal)
    fft_res = abs(fft_res)[:round(len(fft_res) / 2)] / N * 2

    # 2.1 最大值max
    max_fd = np.max(fft_res)

    # 2.2 最小值min
    min_fd = np.min(fft_res)

    # 2.3 均值mean
    mean_fd = np.mean(fft_res)

    # 2.4 中位数mid
    mid_fd = np.median(fft_res)

    # 2.5 方差var
    var_fd = np.var(fft_res)

    # 2.6 均方根rms
    rms_fd = np.sqrt(np.mean(np.square(fft_res)))

    # 2.7 峰值peak
    peak_fd = max(abs(max_fd), abs(min_fd))

    # 2.8 峰峰值peak2peak
    peak2peak_fd = max_fd - min_fd

    # 2.9 标准差std
    std_fd = np.std(fft_res)

    # 2.10 峰值因子crestf
    crestf_fd = max_fd / rms_fd if rms_fd != 0 else 0
    xr = np.square(np.mean(np.sqrt(np.abs(fft_res))))

    # 2.11 裕度margin
    margin_fd = (max_fd / xr) if xr != 0 else 0
    yr = np.mean(np.abs(fft_res))

    # 2.12 脉冲因子pulse
    pulse_fd = max_fd / yr if yr != 0 else 0

    # 2.13 波形因子waveform
    waveform_fd = rms_fd / mean_fd

    # 2.14 峭度因子kur
    kur_fd = scipy.stats.kurtosis(fft_res)

    # 2.15 偏度因子ske
    ske_fd = scipy.stats.skew(fft_res)

    # 2.16 重心频率FC
    ps = fft_res ** 2 / N
    fc_fd = np.sum(fft_res * ps) / np.sum(ps)

    # 2.18 均方根频率
    rmsf_fd = np.sqrt(np.sum(ps * np.square(fft_res)) / np.sum(ps))

    # 2.19 频率方差
    freq_tile = np.tile(fft_res.reshape(1, -1), (1, 1))  # 复制 m 行
    fc_tile = np.tile(fc_fd.reshape(-1, 1), (1, freq_tile.shape[1]))  # 复制 列，与 freq_tile 的列数对应
    vf_fd = np.sum(np.square(freq_tile - fc_tile) * ps) / np.sum(ps)

    # # 2.20 频域矩峰度系数
    # sk_fd = (np.mean(fft_res - mean_fd) ^ 4) / std_fd ^ 4

    # plt.figure(figsize=(8, 4))
    # plt.subplot(2, 1, 1)
    # plt.subplots_adjust(wspace=0, hspace=0.5)
    # plt.plot(signal, linewidth=0.5)
    # plt.title('time domain')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(fft_res, linewidth=0.5)
    # plt.title('FFT')
    # plt.show()

    return (max_fd, min_fd, mean_fd, mid_fd, var_fd,
            rms_fd, peak_fd, peak2peak_fd, std_fd,
            crestf_fd, margin_fd, pulse_fd, waveform_fd,
            kur_fd, ske_fd, fc_fd, rmsf_fd, vf_fd)

# 3. 计算时频域特征
def get_timefreq_domain_feature(data_re, data_im, fs):
    signal = data_re + data_im * 1j
    f, t, zxx = scipy.signal.stft(signal, fs, nperseg=32)
    # zxx = st(signal)
    zxx = abs(zxx)

    zxx_0 = np.sum(zxx[:, 0:12], axis=1)
    zxx_1 = np.sum(zxx[:, 13:25], axis=1)
    zxx_2 = np.sum(zxx[:, 26:38], axis=1)
    zxx_3 = np.sum(zxx[:, 39:51], axis=1)
    zxx_4 = np.sum(zxx[:, 52:63], axis=1)

    # 总能量
    power_tf_sum = np.sum(zxx)

    # 特定频带能量
    power_tf_0 = np.sum(zxx_0) / power_tf_sum
    power_tf_1 = np.sum(zxx_1) / power_tf_sum
    power_tf_2 = np.sum(zxx_2) / power_tf_sum
    power_tf_3 = np.sum(zxx_3) / power_tf_sum
    power_tf_4 = np.sum(zxx_4) / power_tf_sum

    # 谱通量
    flux_tf_0 = np.sum(np.diff(zxx_0) ** 2)
    flux_tf_1 = np.sum(np.diff(zxx_1) ** 2)
    flux_tf_2 = np.sum(np.diff(zxx_2) ** 2)
    flux_tf_3 = np.sum(np.diff(zxx_3) ** 2)
    flux_tf_4 = np.sum(np.diff(zxx_4) ** 2)

    # 谱熵
    entropy_tf_0 = scipy.stats.entropy(zxx_0) if np.sum(zxx_0) != 0 else 0
    entropy_tf_1 = scipy.stats.entropy(zxx_1) if np.sum(zxx_1) != 0 else 0
    entropy_tf_2 = scipy.stats.entropy(zxx_2) if np.sum(zxx_2) != 0 else 0
    entropy_tf_3 = scipy.stats.entropy(zxx_3) if np.sum(zxx_3) != 0 else 0
    entropy_tf_4 = scipy.stats.entropy(zxx_4) if np.sum(zxx_4) != 0 else 0

    # 谱平坦度
    flatness_tf_0 = scipy.stats.gmean(zxx_0) / np.mean(zxx_0) if np.mean(zxx_0) != 0 else 0
    flatness_tf_1 = scipy.stats.gmean(zxx_1) / np.mean(zxx_1) if np.mean(zxx_1) != 0 else 0
    flatness_tf_2 = scipy.stats.gmean(zxx_2) / np.mean(zxx_2) if np.mean(zxx_2) != 0 else 0
    flatness_tf_3 = scipy.stats.gmean(zxx_3) / np.mean(zxx_3) if np.mean(zxx_3) != 0 else 0
    flatness_tf_4 = scipy.stats.gmean(zxx_4) / np.mean(zxx_4) if np.mean(zxx_4) != 0 else 0

    # pca
    pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(zxx)
    pca.transform(zxx)
    pca_0, pca_1, pca_2 = pca.explained_variance_ratio_

    # plt.pcolormesh(t, f, np.abs(zxx))
    # plt.colorbar()
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.tight_layout()
    # plt.show()

    return (power_tf_sum, pca_0, pca_1, pca_2,
            power_tf_0, power_tf_1, power_tf_2, power_tf_3, power_tf_4,
            flux_tf_0, flux_tf_1, flux_tf_2, flux_tf_3, flux_tf_4,
            entropy_tf_0, entropy_tf_1, entropy_tf_2, entropy_tf_3, entropy_tf_4,
            flatness_tf_0, flatness_tf_1, flatness_tf_2, flatness_tf_3, flatness_tf_4)

# 4. 计算相位域特征
def get_phase_domain_feature(data_re, data_im):
    signal = data_re + data_im * 1j
    phase = np.angle(signal)

    # 2.1 最大值max
    max_pd = np.max(phase)

    # 2.2 最小值min
    min_pd = np.min(phase)

    # 2.3 均值mean
    mean_pd = np.mean(phase)

    # 2.4 中位数mid
    mid_pd = np.median(phase)

    # 2.5 方差var
    var_pd = np.var(phase)

    # 2.6 均方根rms
    rms_pd = np.sqrt(np.mean(np.square(phase)))

    # 2.7 峰值peak
    peak_pd = max(abs(max_pd), abs(min_pd))

    # 2.8 峰峰值peak2peak
    peak2peak_pd = max_pd - min_pd

    # 2.9 标准差std
    std_pd = np.std(phase)

    # 2.10 峰值因子crestf
    crestf_pd = max_pd / rms_pd if rms_pd != 0 else 0
    xr = np.square(np.mean(np.sqrt(np.abs(phase))))

    # 2.11 裕度margin
    margin_pd = (max_pd / xr) if xr != 0 else 0
    yr = np.mean(np.abs(phase))

    # 2.12 脉冲因子pulse
    pulse_pd = max_pd / yr if yr != 0 else 0

    # 2.13 波形因子waveform
    waveform_pd = rms_pd / mean_pd

    # 2.14 峭度因子kur
    kur_pd = scipy.stats.kurtosis(phase)

    # 2.15 偏度因子ske
    ske_pd = scipy.stats.skew(phase)

    return (max_pd, min_pd, mean_pd, mid_pd, var_pd,
            rms_pd, peak_pd, peak2peak_pd, std_pd,
            crestf_pd, margin_pd, pulse_pd, waveform_pd,
            kur_pd, ske_pd)

def proportional_extraction_data(data_path, copy_path, num_total, proportion_list=[0.69, 0.05, 0.05, 0.04, 0.05, 0.01, 0.01, 0.05, 0.05], random_state=0):
    warnings.filterwarnings("ignore", category=UserWarning)
    # 读取数据
    # 读取数据
    featuremap_path = r'E:\03-dataset\output\prediction_20231010.csv'
    featuremap = pd.read_csv(featuremap_path)

    type_list_0 = featuremap.loc[featuremap['type_num'] == 0]['num'].reset_index(drop=True)
    type_list_1 = featuremap.loc[featuremap['type_num'] == 1]['num'].reset_index(drop=True)
    type_list_2 = featuremap.loc[featuremap['type_num'] == 2]['num'].reset_index(drop=True)
    type_list_3 = featuremap.loc[featuremap['type_num'] == 3]['num'].reset_index(drop=True)
    type_list_4 = featuremap.loc[featuremap['type_num'] == 4]['num'].reset_index(drop=True)
    type_list_5 = featuremap.loc[featuremap['type_num'] == 5]['num'].reset_index(drop=True)
    type_list_6 = featuremap.loc[featuremap['type_num'] == 6]['num'].reset_index(drop=True)
    type_list_7 = featuremap.loc[featuremap['type_num'] == 7]['num'].reset_index(drop=True)
    type_list_8 = featuremap.loc[featuremap['type_num'] == 8]['num'].reset_index(drop=True)

    data_list_0 = type_list_0.sample(n=round(num_total * proportion_list[0]), random_state=random_state, axis=0).reset_index(drop=True)
    data_list_1 = type_list_1.sample(n=round(num_total * proportion_list[1]), random_state=random_state, axis=0).reset_index(drop=True)
    data_list_2 = type_list_2.sample(n=round(num_total * proportion_list[2]), random_state=random_state, axis=0).reset_index(drop=True)
    data_list_3 = type_list_3.sample(n=round(num_total * proportion_list[3]), random_state=random_state, axis=0).reset_index(drop=True)
    data_list_4 = type_list_4.sample(n=round(num_total * proportion_list[4]), random_state=random_state, axis=0).reset_index(drop=True)
    data_list_5 = type_list_5.sample(n=round(num_total * proportion_list[5]), random_state=random_state, axis=0).reset_index(drop=True)
    data_list_6 = type_list_6.sample(n=round(num_total * proportion_list[6]), random_state=random_state, axis=0).reset_index(drop=True)
    data_list_7 = type_list_7.sample(n=round(num_total * proportion_list[7]), random_state=random_state, axis=0).reset_index(drop=True)
    data_list_8 = type_list_8.sample(n=round(num_total * proportion_list[8]), random_state=random_state, axis=0).reset_index(drop=True)

    for i in range(0, len(data_list_0)):
        shutil.copyfile(os.path.join(data_path, 'S{}.csv'.format(data_list_0.loc[i])),
                        os.path.join(copy_path, 'S{}.csv'.format(data_list_0[i])))
        shutil.copyfile(os.path.join(data_path, 'S{}.hea'.format(data_list_0[i])),
                        os.path.join(copy_path, 'S{}.hea'.format(data_list_0[i])))

    for i in range(0, len(data_list_1)):
        shutil.copyfile(os.path.join(data_path, 'S{}.csv'.format(data_list_1[i])),
                        os.path.join(copy_path, 'S{}.csv'.format(data_list_1[i])))
        shutil.copyfile(os.path.join(data_path, 'S{}.hea'.format(data_list_1[i])),
                        os.path.join(copy_path, 'S{}.hea'.format(data_list_1[i])))

    for i in range(0, len(data_list_2)):
        shutil.copyfile(os.path.join(data_path, 'S{}.csv'.format(data_list_2[i])),
                        os.path.join(copy_path, 'S{}.csv'.format(data_list_2[i])))
        shutil.copyfile(os.path.join(data_path, 'S{}.hea'.format(data_list_2[i])),
                        os.path.join(copy_path, 'S{}.hea'.format(data_list_2[i])))

    for i in range(0, len(data_list_3)):
        shutil.copyfile(os.path.join(data_path, 'S{}.csv'.format(data_list_3[i])),
                        os.path.join(copy_path, 'S{}.csv'.format(data_list_3[i])))
        shutil.copyfile(os.path.join(data_path, 'S{}.hea'.format(data_list_3[i])),
                        os.path.join(copy_path, 'S{}.hea'.format(data_list_3[i])))

    for i in range(0, len(data_list_4)):
        shutil.copyfile(os.path.join(data_path, 'S{}.csv'.format(data_list_4[i])),
                        os.path.join(copy_path, 'S{}.csv'.format(data_list_4[i])))
        shutil.copyfile(os.path.join(data_path, 'S{}.hea'.format(data_list_4[i])),
                        os.path.join(copy_path, 'S{}.hea'.format(data_list_4[i])))

    for i in range(0, len(data_list_5)):
        shutil.copyfile(os.path.join(data_path, 'S{}.csv'.format(data_list_5[i])),
                        os.path.join(copy_path, 'S{}.csv'.format(data_list_5[i])))
        shutil.copyfile(os.path.join(data_path, 'S{}.hea'.format(data_list_5[i])),
                        os.path.join(copy_path, 'S{}.hea'.format(data_list_5[i])))

    for i in range(0, len(data_list_6)):
        shutil.copyfile(os.path.join(data_path, 'S{}.csv'.format(data_list_6[i])),
                        os.path.join(copy_path, 'S{}.csv'.format(data_list_6[i])))
        shutil.copyfile(os.path.join(data_path, 'S{}.hea'.format(data_list_6[i])),
                        os.path.join(copy_path, 'S{}.hea'.format(data_list_6[i])))

    for i in range(0, len(data_list_7)):
        shutil.copyfile(os.path.join(data_path, 'S{}.csv'.format(data_list_7[i])),
                        os.path.join(copy_path, 'S{}.csv'.format(data_list_7[i])))
        shutil.copyfile(os.path.join(data_path, 'S{}.hea'.format(data_list_7[i])),
                        os.path.join(copy_path, 'S{}.hea'.format(data_list_7[i])))

    for i in range(0, len(data_list_8)):
        shutil.copyfile(os.path.join(data_path, 'S{}.csv'.format(data_list_8[i])),
                        os.path.join(copy_path, 'S{}.csv'.format(data_list_8[i])))
        shutil.copyfile(os.path.join(data_path, 'S{}.hea'.format(data_list_8[i])),
                        os.path.join(copy_path, 'S{}.hea'.format(data_list_8[i])))

def create_feature_map(data_path, out_path, N=1000, fs=100e6):
    warnings.filterwarnings("ignore", category=UserWarning)
    files = os.listdir(data_path)
    output = []

    for f in files:
        data_name = f.split('.')[0]
        suffix = f.split('.')[1]
        if suffix == 'csv':
            print('processing {}'.format(data_name))
            # 1. 读取数据
            csv_path = r'{}\{}.csv'.format(data_path, data_name)  # csv地址
            hea_path = r'{}\{}.hea'.format(data_path, data_name)  # csv地址

            data = pd.read_csv(csv_path, header=None)
            data_re = data.loc[:, 0]  # 数据的实部
            data_im = data.loc[:, 1]  # 数据的虚部

            label = pd.read_csv(hea_path, header=None)
            type = label.loc[0, 0]
            if type == 'normal':
                type_num = 0
            elif type == 'frequency converte':
                type_num = 1
            elif type == 'transient':
                type_num = 2
            elif type == 'interuption':
                type_num = 3
            elif type == 'harmonic':
                type_num = 4
            elif type == 'meteor':
                type_num = 5
            elif type == 'flash':
                type_num = 6
            elif type == 'cross-modulation':
                type_num = 7
            elif type == 'inter-modulation':
                type_num = 8
            else:
                type_num = 99

            # 2. 计算特征
            # 2.1 时域特征
            (mean_re_td, mean_im_td, mid_re_td, mid_im_td,
             var_re_td, var_im_td, rms_re_td, rms_im_td,
             std_re_td, std_im_td, waveform_re_td, waveform_im_td,
             kur_re_td, kur_im_td, ske_re_td, ske_im_td) = tool_calculate.get_time_domain_features(data_re, data_im)

            # 2.2 频域特征
            (max_fd, min_fd, mean_fd, mid_fd, var_fd,
             rms_fd, peak_fd, peak2peak_fd, std_fd,
             crestf_fd, margin_fd, pulse_fd, waveform_fd,
             kur_fd, ske_fd, fc_fd, rmsf_fd, vf_fd) = tool_calculate.get_frequncy_domain_feature(data_re, data_im, N, fs)

            # 2.3 时频域特征
            (power_tf_sum, pca_0, pca_1, pca_2,
             power_tf_0, power_tf_1, power_tf_2, power_tf_3, power_tf_4,
             flux_tf_0, flux_tf_1, flux_tf_2, flux_tf_3, flux_tf_4,
             entropy_tf_0, entropy_tf_1, entropy_tf_2, entropy_tf_3, entropy_tf_4,
             flatness_tf_0, flatness_tf_1, flatness_tf_2, flatness_tf_3,
             flatness_tf_4) = tool_calculate.get_timefreq_domain_feature(data_re, data_im, fs)

            # 2.4 相位谱
            (max_pd, min_pd, mean_pd, mid_pd, var_pd,
             rms_pd, peak_pd, peak2peak_pd, std_pd,
             crestf_pd, margin_pd, pulse_pd, waveform_pd,
             kur_pd, ske_pd) = tool_calculate.get_phase_domain_feature(data_re, data_im)

            output.append(np.float32([type_num, mean_re_td, mean_im_td, mid_re_td, mid_im_td,  # time_domain
                                      var_re_td, var_im_td, rms_re_td, rms_im_td,
                                      std_re_td, std_im_td, waveform_re_td, waveform_im_td,
                                      kur_re_td, kur_im_td, ske_re_td, ske_im_td,

                                      max_fd, min_fd, mean_fd, mid_fd, var_fd,  # freq_domain
                                      rms_fd, peak_fd, peak2peak_fd, std_fd,
                                      crestf_fd, margin_fd, pulse_fd, waveform_fd,
                                      kur_fd, ske_fd, fc_fd, rmsf_fd, vf_fd,

                                      power_tf_sum, pca_0, pca_1, pca_2,  # tf_domain
                                      power_tf_0, power_tf_1, power_tf_2, power_tf_3, power_tf_4,
                                      flux_tf_0, flux_tf_1, flux_tf_2, flux_tf_3, flux_tf_4,
                                      entropy_tf_0, entropy_tf_1, entropy_tf_2, entropy_tf_3, entropy_tf_4,
                                      flatness_tf_0, flatness_tf_1, flatness_tf_2, flatness_tf_3, flatness_tf_4,

                                      max_pd, min_pd, mean_pd, mid_pd, var_pd,  # phase_domain
                                      rms_pd, peak_pd, peak2peak_pd, std_pd,
                                      crestf_pd, margin_pd, pulse_pd, waveform_pd,
                                      kur_pd, ske_pd]))

        # 输出为csv格式
        df_dataframe = pd.DataFrame(output, columns=['type_num', 'mean_re_td', 'mean_im_td', 'mid_re_td', 'mid_im_td',
                                                     # time_domain
                                                     'var_re_td', 'var_im_td', 'rms_re_td', 'rms_im_td',
                                                     'std_re_td', 'std_im_td', 'waveform_re_td', 'waveform_im_td',
                                                     'kur_re_td', 'kur_im_td', 'ske_re_td', 'ske_im_td',

                                                     'max_fd', 'min_fd', 'mean_fd', 'mid_fd', 'var_fd',  # freq_domain
                                                     'rms_fd', 'peak_fd', 'peak2peak_fd', 'std_fd',
                                                     'crestf_fd', 'margin_fd', 'pulse_fd', 'waveform_fd',
                                                     'kur_fd', 'ske_fd', 'fc_fd', 'rmsf_fd', 'vf_fd',

                                                     'power_tf_sum', 'pca_0', 'pca_1', 'pca_2',  # tf_domain
                                                     'power_tf_0', 'power_tf_1', 'power_tf_2', 'power_tf_3', 'power_tf_4',
                                                     'flux_tf_0', 'flux_tf_1', 'flux_tf_2', 'flux_tf_3', 'flux_tf_4',
                                                     'entropy_tf_0', 'entropy_tf_1', 'entropy_tf_2', 'entropy_tf_3',
                                                     'entropy_tf_4',
                                                     'flatness_tf_0', 'flatness_tf_1', 'flatness_tf_2', 'flatness_tf_3',
                                                     'flatness_tf_4',

                                                     'max_pd', 'min_pd', 'mean_pd', 'mid_pd', 'var_pd',  # phase_domain
                                                     'rms_pd', 'peak_pd', 'peak2peak_pd', 'std_pd',
                                                     'crestf_pd', 'margin_pd', 'pulse_pd', 'waveform_pd',
                                                     'kur_pd', 'ske_pd'])

    df_dataframe.to_csv(out_path, index=False, encoding='GBK')

def predict_result(model_path, featuremap_path):
    warnings.filterwarnings("ignore", category=UserWarning)
    model = joblib.load(model_path)
    # 读取数据
    featuremap = pd.read_csv(featuremap_path)
    X = featuremap.drop("type_num", axis=1)
    Y = featuremap["type_num"]
    result = model.predict(X)
    score = accuracy_score(Y, result)
    return Y, result, score


def plot_confusion_matrix(True_label, T_predict1):
    # True_label = result['type_num']
    # T_predict1 = result['prediction']
    C1 = confusion_matrix(True_label, T_predict1,
                          normalize='true')  # True_label 真实标签 shape=(n,1);T_predict1 预测标签 shape=(n,1)

    xtick = ['正常', '捷变频', '瞬态', '信号中断', '谐波干扰', '流星余迹', '闪电干扰', '交调', '互调']
    ytick = ['正常', '捷变频', '瞬态', '信号中断', '谐波干扰', '流星余迹', '闪电干扰', '交调', '互调']
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    sns.heatmap(C1,
                fmt='g',
                cmap=sns.diverging_palette(20, 220, n=200),
                annot=True,
                cbar=True,
                square=True,
                xticklabels=xtick,
                yticklabels=ytick)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒

def plot_time_domain(data_name, base_path, N=1000, f=100e6):
    data_path = os.path.join(base_path, '{}.csv'.format(data_name))
    y = pd.read_csv(data_path, header=None)
    re = y.iloc[:, 0]
    im = y.iloc[:, 1]
    complex = re + 1j * im
    data = abs(complex)/max(abs(complex))
    x = np.linspace(0, N / f, N)

    # 时域图
    plt.figure()
    plt.plot(x, data)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title('{}信号时域图'.format(data_name))
    plt.ylabel('归一化幅值')
    plt.xlabel('时间(s)')

def plot_frequency_domain(data_name, base_path, N=1000, f=100e6):
    data_path = os.path.join(base_path, '{}.csv'.format(data_name))
    y = pd.read_csv(data_path, header=None)
    re = y.iloc[:, 0]
    im = y.iloc[:, 1]
    complex = re + 1j * im
    x = np.linspace(0, N / f, N)

    fft_res = np.fft.fft(complex)
    fft_res = abs(fft_res)[:round(len(fft_res) / 2)] / N * 2
    list1 = np.array(range(0, int(N / 2)))
    freq1 = f * list1 / N

    plt.figure()
    plt.plot(freq1, fft_res)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title('{}信号频域图'.format(data_name))
    plt.ylabel('幅值')
    plt.xlabel('频率(Hz)')


def plot_tf_domain(data_name, base_path, N=1000, f=100e6):
    data_path = os.path.join(base_path, '{}.csv'.format(data_name))
    y = pd.read_csv(data_path, header=None)
    re = y.iloc[:, 0]
    im = y.iloc[:, 1]
    complex = re + 1j * im
    x = np.linspace(0, N / f, N)

    zxx = st(complex)
    list2 = np.array(range(0, int(N)))
    freq2 = f * list2 / N
    plt.figure()
    plt.pcolormesh(x, freq2, np.abs(zxx))
    plt.title('{}信号时频域图'.format(data_name))
    plt.ylabel('频率(Hz)')
    plt.xlabel('时间(s)')
    plt.tight_layout()

def plot_phase_domain(data_name, base_path, N=1000, f=100e6):
    data_path = os.path.join(base_path, '{}.csv'.format(data_name))
    y = pd.read_csv(data_path, header=None)
    re = y.iloc[:, 0]
    im = y.iloc[:, 1]
    complex = re + 1j * im
    x = np.linspace(0, N / f, N)

    phase = np.angle(complex)
    plt.figure()
    plt.plot(x, phase)
    plt.ylabel('相角')
    plt.xlabel('时间(s)')
    plt.title('{}信号相位域图'.format(data_name))
    plt.tight_layout()
    plt.show()

def predict_single_sig(data_name, base_path, N=1000, fs=100e6):
    warnings.filterwarnings("ignore", category=UserWarning)
    model = joblib.load(model_path)

    # 1. 读取数据
    csv_path = r'{}\{}.csv'.format(base_path, data_name)  # csv地址
    hea_path = r'{}\{}.hea'.format(base_path, data_name)  # csv地址

    data = pd.read_csv(csv_path, header=None)
    data_re = data.loc[:, 0]  # 数据的实部
    data_im = data.loc[:, 1]  # 数据的虚部

    label = pd.read_csv(hea_path, header=None)
    type = label.loc[0, 0]
    if type == 'normal':
        type_num = 0
    elif type == 'frequency converte':
        type_num = 1
    elif type == 'transient':
        type_num = 2
    elif type == 'interuption':
        type_num = 3
    elif type == 'harmonic':
        type_num = 4
    elif type == 'meteor':
        type_num = 5
    elif type == 'flash':
        type_num = 6
    elif type == 'cross-modulation':
        type_num = 7
    elif type == 'inter-modulation':
        type_num = 8
    else:
        type_num = 99

    # 2. 计算特征
    # 2.1 时域特征
    (mean_re_td, mean_im_td, mid_re_td, mid_im_td,
     var_re_td, var_im_td, rms_re_td, rms_im_td,
     std_re_td, std_im_td, waveform_re_td, waveform_im_td,
     kur_re_td, kur_im_td, ske_re_td, ske_im_td) = tool_calculate.get_time_domain_features(data_re, data_im)

    # 2.2 频域特征
    (max_fd, min_fd, mean_fd, mid_fd, var_fd,
     rms_fd, peak_fd, peak2peak_fd, std_fd,
     crestf_fd, margin_fd, pulse_fd, waveform_fd,
     kur_fd, ske_fd, fc_fd, rmsf_fd, vf_fd) = tool_calculate.get_frequncy_domain_feature(data_re, data_im, N, fs)

    # 2.3 时频域特征
    (power_tf_sum, pca_0, pca_1, pca_2,
     power_tf_0, power_tf_1, power_tf_2, power_tf_3, power_tf_4,
     flux_tf_0, flux_tf_1, flux_tf_2, flux_tf_3, flux_tf_4,
     entropy_tf_0, entropy_tf_1, entropy_tf_2, entropy_tf_3, entropy_tf_4,
     flatness_tf_0, flatness_tf_1, flatness_tf_2, flatness_tf_3,
     flatness_tf_4) = tool_calculate.get_timefreq_domain_feature(data_re, data_im, fs)

    # 2.4 相位谱
    (max_pd, min_pd, mean_pd, mid_pd, var_pd,
     rms_pd, peak_pd, peak2peak_pd, std_pd,
     crestf_pd, margin_pd, pulse_pd, waveform_pd,
     kur_pd, ske_pd) = tool_calculate.get_phase_domain_feature(data_re, data_im)
    output = []
    output.append(np.float32([type_num, mean_re_td, mean_im_td, mid_re_td, mid_im_td,  # time_domain
                              var_re_td, var_im_td, rms_re_td, rms_im_td,
                              std_re_td, std_im_td, waveform_re_td, waveform_im_td,
                              kur_re_td, kur_im_td, ske_re_td, ske_im_td,

                              max_fd, min_fd, mean_fd, mid_fd, var_fd,  # freq_domain
                              rms_fd, peak_fd, peak2peak_fd, std_fd,
                              crestf_fd, margin_fd, pulse_fd, waveform_fd,
                              kur_fd, ske_fd, fc_fd, rmsf_fd, vf_fd,

                              power_tf_sum, pca_0, pca_1, pca_2,  # tf_domain
                              power_tf_0, power_tf_1, power_tf_2, power_tf_3, power_tf_4,
                              flux_tf_0, flux_tf_1, flux_tf_2, flux_tf_3, flux_tf_4,
                              entropy_tf_0, entropy_tf_1, entropy_tf_2, entropy_tf_3, entropy_tf_4,
                              flatness_tf_0, flatness_tf_1, flatness_tf_2, flatness_tf_3, flatness_tf_4,

                              max_pd, min_pd, mean_pd, mid_pd, var_pd,  # phase_domain
                              rms_pd, peak_pd, peak2peak_pd, std_pd,
                              crestf_pd, margin_pd, pulse_pd, waveform_pd,
                              kur_pd, ske_pd]))

    # 输出为csv格式
    featuremap = pd.DataFrame(output, columns=['type_num', 'mean_re_td', 'mean_im_td', 'mid_re_td', 'mid_im_td',
                                                 # time_domain
                                                 'var_re_td', 'var_im_td', 'rms_re_td', 'rms_im_td',
                                                 'std_re_td', 'std_im_td', 'waveform_re_td', 'waveform_im_td',
                                                 'kur_re_td', 'kur_im_td', 'ske_re_td', 'ske_im_td',

                                                 'max_fd', 'min_fd', 'mean_fd', 'mid_fd', 'var_fd',  # freq_domain
                                                 'rms_fd', 'peak_fd', 'peak2peak_fd', 'std_fd',
                                                 'crestf_fd', 'margin_fd', 'pulse_fd', 'waveform_fd',
                                                 'kur_fd', 'ske_fd', 'fc_fd', 'rmsf_fd', 'vf_fd',

                                                 'power_tf_sum', 'pca_0', 'pca_1', 'pca_2',  # tf_domain
                                                 'power_tf_0', 'power_tf_1', 'power_tf_2', 'power_tf_3', 'power_tf_4',
                                                 'flux_tf_0', 'flux_tf_1', 'flux_tf_2', 'flux_tf_3', 'flux_tf_4',
                                                 'entropy_tf_0', 'entropy_tf_1', 'entropy_tf_2', 'entropy_tf_3',
                                                 'entropy_tf_4',
                                                 'flatness_tf_0', 'flatness_tf_1', 'flatness_tf_2', 'flatness_tf_3',
                                                 'flatness_tf_4',

                                                 'max_pd', 'min_pd', 'mean_pd', 'mid_pd', 'var_pd',  # phase_domain
                                                 'rms_pd', 'peak_pd', 'peak2peak_pd', 'std_pd',
                                                 'crestf_pd', 'margin_pd', 'pulse_pd', 'waveform_pd',
                                                 'kur_pd', 'ske_pd'])

    X = featuremap.drop("type_num", axis=1)
    predict = model.predict(X)
    if predict == 0:
        predict_single = 'normal'
    elif predict == 1:
        predict_single = 'frequency converte'
    elif predict == 2:
        predict_single = 'transient'
    elif predict == 3:
        predict_single = 'interuption'
    elif predict == 4:
        predict_single = 'harmonic'
    elif predict == 5:
        predict_single = 'meteor'
    elif predict == 6:
        predict_single = 'flash'
    elif predict == 7:
        predict_single = 'cross-modulation'
    elif predict == 8:
        predict_single = 'inter-modulation'


    return type, predict_single



if __name__ == '__main__':
    data_path = r'E:\BLS\03-dataset\selfmake_dataset'
    copy_path = r'E:\BLS\03-dataset\old_dataset\500'
    out_path = r'E:\BLS\03-dataset\feature_20231025_extract.csv'
    model_path = r'E:\BLS\03-dataset\old_dataset\dump_model_20231025.pkl'
    num_total = 500
    proportion_list = [0.69, 0.05, 0.05, 0.04, 0.05, 0.01, 0.01, 0.05, 0.05]
    sum = sum(proportion_list)
    random_state = 0
    data_name = 'S10000'
    N = 1000
    f = 100e6

    # 按比例抽取数据形成数据库
    proportional_extraction_data(data_path, copy_path, num_total, proportion_list, random_state)
    # 计算特征矩阵
    create_feature_map(copy_path, out_path)
    # 读取模型，输出预测结果
    True_label, T_predict, score = predict_result(model_path, out_path)
    print(score)
    # 绘制混淆矩阵图
    plot_confusion_matrix(True_label, T_predict)
    # 单条数据结果预测
    label_single, predict_single = predict_single_sig(data_name, data_path)
    print('True_label:{}'.format(label_single))
    print('T_predict:{}'.format(predict_single))
    # 绘制时域图
    plot_time_domain(data_name, data_path)
    # 绘制频域图
    plot_frequency_domain(data_name, data_path)
    # 绘制时频域图
    plot_tf_domain(data_name, data_path)
    # 绘制相位域图
    plot_phase_domain(data_name, data_path)
    plt.show()
