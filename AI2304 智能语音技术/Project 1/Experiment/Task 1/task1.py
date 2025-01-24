import evaluate
import vad_utils
import matplotlib.pyplot as plt
import numpy as np
import wave
import os

#计算短时能量
def compute_energy(signal):
    energy = 0
    for i in range(len(signal)):
        energy += signal[i]*signal[i]
    return energy

#用于在test(作业要求)数据上做检测,使用了短时能量
def process_test(f):
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    wlen = int(0.032 * framerate)
    inc = int(0.010 * framerate)
    str_data = f.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data = wave_data*1.0/(max(abs(wave_data)))
    time = np.arange(0, wlen) * (1.0 / framerate)
    signal_length = len(wave_data)
    nf = int(np.ceil((1.0*signal_length-wlen+inc)/inc))
    pad_length = int((nf-1)*inc+wlen) 
    zeros = np.zeros((pad_length-signal_length,)) 
    pad_signal = np.concatenate((wave_data,zeros))
    indices = np.tile(np.arange(0,wlen),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(wlen,1)).T
    indices = np.array(indices,dtype=np.int32)
    frames = pad_signal[indices]
    energyres = []

    for i in range(nf):
        a = frames[i:i+1]
        windown=np.hanning(wlen)  #调用汉明窗
        b=a[0]*windown
        res_energy = compute_energy(b)
        energyres.append(res_energy)

    while len(energyres) < nf:
        energyres.append(0)

    res = vad_utils.prediction_to_vad_label(energyres, 0.032, 0.010, 0.06) #以0.06为阈值做分类
    energyres = vad_utils.parse_vad_label(res,0.032,0.010) #得到分类的标签结果
    silence = 0
    flag = 0
    start = 1
    for i in range(len(energyres)):
        if energyres[i] == 1 & start == 1:
            start = 0
        elif energyres[i] == 0 & start == 0:
            flag = 1
            silence += 1
        elif energyres[i] == 1 & flag == 1:
            flag = 0
            if silence <= 2:
                for j in range(silence):
                    energyres[i-j-1] = 1
            silence = 0

    res = vad_utils.prediction_to_vad_label(energyres, 0.032, 0.010, 0.06) #以0.06为阈值做分类
    return res #返回string变量(例如:"0.02,0.1 0.15,0.17, 0.29,0.40")

if __name__ == '__main__':
    f_res = open("./result3.txt","w") #写入结果的文件
    f_res.truncate(0) #清空文件
    paths = os.walk(r"./wavs/test") #得到路径
    for path, dir_lst, file_lst in paths:
        for file_name in file_lst: #遍历测试集的音频
            file_name = file_name.split(".")[0] #分离出文件名(不带后缀)
            f = wave.open(r"./wavs/test/{name}.wav".format(name = file_name), "rb") #读取音频
            res = process_test(f) #进行处理
            f_res.write("{name} {result}".format(name = file_name, result = res)+"\n") #将结果写入result.txt(音频按名称已排序)
    f_res.close() #关闭文件