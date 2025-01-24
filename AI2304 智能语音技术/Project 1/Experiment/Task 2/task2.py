import evaluate
import vad_utils
import numpy as np
import wave
import scipy.io.wavfile
from python_speech_features import mfcc
from python_speech_features import logfbank
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
if __name__ == '__main__':
    vad_label = vad_utils.read_label_from_file("./data/train_label.txt") #读取字典
    flag = True
    train_all_label = []
    for key in vad_label: #遍历
        fs, audio = scipy.io.wavfile.read("./wavs/train/{name}.wav".format(name = key)) #读取音频    
        train_single_mfcc = mfcc(audio, samplerate=fs, winlen=0.032, winstep=0.010) #提取mfcc
        train_single_logfbank = logfbank(audio, samplerate=fs, winlen=0.032, winstep=0.010)
        train_single_label = vad_label[key] #标签
        while len(vad_label[key]) < train_single_mfcc.shape[0]:
            vad_label[key].append(0) #补齐
        if flag == True:
            train_all_mfcc = train_single_mfcc
            train_all_logfbank = train_single_logfbank
            flag = False
        else:
            train_all_logfbank = np.vstack((train_all_logfbank,train_single_logfbank))
            train_all_mfcc = np.vstack((train_all_mfcc, train_single_mfcc)) #合成总mfcc特征集
        train_all_label.extend(vad_label[key]) #合成总特征集
        print('finish: {name}'.format(name = key))
    print(train_all_logfbank.shape)
    print(train_all_mfcc.shape)
    print(len(train_all_label))
    gmm = GaussianMixture(n_components=2, covariance_type='full', max_iter=3000)
    gmm.fit(train_all_mfcc)
    f_res = open("./result2.txt","w") #写入结果的文件
    f_res.truncate(0) #清空文件
    paths = os.walk(r"./wavs/test") #得到路径
    for path, dir_lst, file_lst in paths:
        for file_name in file_lst: #遍历测试集的音频
            file_name = file_name.split(".")[0] #分离出文件名(不带后缀)
            fs, audio = scipy.io.wavfile.read("./wavs/test/{name}.wav".format(name = file_name))
            test_mfcc = mfcc(audio, samplerate=fs, winlen=0.032, winstep=0.010)
            prediction = gmm.predict(test_mfcc)
            res = vad_utils.prediction_to_vad_label(prediction=prediction, threshold=0.5)
            f_res.write("{name} {result}".format(name = file_name, result = res)+"\n") #将结果写入result.txt(音频按名称已排序)
    f_res.close() #关闭文件