import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt

from librosa import load
from librosa.feature import melspectrogram


class OpenAudioFile():
    """
    Open wav file
    """
    def __init__(self, rate=44100, loader='librosa'):
        self.loader = loader
        self.rate = rate

    def __call__(self, fname):
        file_name = self.directory + fname

        if self.loader == 'librosa':
            audio, rate = load(file_name, sr=self.rate)
        elif self.loader == 'scipy':
            rate, audio = scipy.io.wavfile.read(file_name)
        else:
            raise ValueError("unknown audio loader; loader must be ['librosa', 'scipy']")

        return audio


class OpenMelSpectrFile():
    """
    Open .npy mel spectr file
    """
    def __init__(self, suffix='_spectr'):
        self.suffix = suffix
        self.valid = True

    def __call__(self, fname):
        file_name = self.directory.strip("/") + self.suffix + "/" + fname[:-3] + "npy"
        mel_spectr = np.load(file_name)
        return mel_spectr


class Cut():
    def __init__(self, window):
        self.window = window
        self.valid = True

    def __call__(self, data):
        return data[:, :self.window]


class ZeroPad():
    def __init__(self, window):
        self.window = window
        self.valid = True

    def __call__(self, data):
        if data.shape[1] > self.window:
            return data
        return np.pad(data, ((0, 0), (0, self.window - data.shape[1])), 'constant', constant_values=(0,))


class Normalize():
    def __init__(self, channel=-1):
        self.channel = channel
        self.valid = True

    def __call__(self, data):
        if self.channel == -1:
            if data.std() > 0.0:
                return (data - data.mean()) / data.std()
        else:
            if data[self.channel].std() > 0.0:
                data[self.channel] = (data[self.channel] - data[self.channel].mean()) / data[self.channel].std()
        return data


class NormalizeHorizontal():
    def __init__(self):
        self.valid = True

    def __call__(self, data):
        ans = (data.T - data.min(axis=1)) / (data.max(axis=1) - data.min(axis=1) + 0.0000001)
        return ans.T


class NormalizeAudio():
    def __init__(self):
        self.valid = True

    def __call__(self, data):
        return (data - data.mean()) / (data.std() + 0.0001)


class TrimAudio():
    def __init__(self, level):
        self.level = level
        self.valid = True

    def __call__(self, data):
        level = self.level * abs(data).max()
        left = 0
        while left < data.shape[0] and data[left] < level:
            left += 1
        right = data.shape[0] - 1
        while right >= 0 and data[right] < level:
            right -= 1
        return data[left:right]


class MinMaxScale():
    def __init__(self):
        self.valid = True

    def __call__(self, data):
        if data.std() - data.min() > 0.0:
            return (data - data.min()) / (data.max() - data.min())
        return data


class RandomShiftWithZeroPad():
    def __init__(self, window):
        self.valid = True

    def __call__(self, data):
        if data.shape[1] >= self.window:
            return data
        left = np.random.randint(self.window - data.shape[1])
        right = self.window - data.shape[1] - left
        return np.pad(data, ((0, 0), (left, right)), 'constant', constant_values=(0,))


class RandomPitch():
    def __init__(self, pixels, validation=True):
        self.pixels = pixels
        self.valid = validation

    def __call__(self, data):
        h = np.random.randint(-self.pixels, self.pixels)
        if h == 0:
            return data
        minvalue = np.mean(data)
        if h > 0:
            data[:-h] = data[h:]
            data[-h:] = minvalue * np.ones((h, data.shape[1]))
        if h < 0:
            h = -h
            data[:h] = minvalue * np.ones((h, data.shape[1]))
            data[h:] = data[:-h]
        return data


class Log():
    def __init__(self, margin, zero):
        self.margin = margin
        self.zero = zero
        self.valid = True

    def __call__(self, data):
        data[data < self.margin] = self.zero
        return np.log(data)


class Power():
    def __init__(self, power=0.125):
        self.power = power
        self.valid = True

    def __call__(self, data):
        return data ** self.power


class DeltaTime():
    def __init__(self, channel, inplace):
        self.channel = channel
        self.inplace = bool(inplace)
        self.valid = True

    def __call__(self, data):
        old_channel = data[self.channel]
        new_channel = ZeroPad(data.shape[-1])(old_channel[:, 1:] - old_channel[:, :-1])
        if self.inplace:
            data[self.channel] = new_channel
            return data
        return np.concatenate([data, np.expand_dims(new_channel, axis=0)], axis=0).astype('float32')


class DeltaDeltaTime():
    def __init__(self):
        self.valid = True

    def __call__(self, data):
        return DeltaTime()(data)


class Trim():
    def __init__(self, level, func='mean'):
        self.level = level
        self.func = func
        self.valid = True

    def __call__(self, data):
        mx = getattr(np, self.func)(data, axis=0)
        level = self.level * mx.max()
        left = 0
        while left < data.shape[1] and mx[left] < level:
            left += 1
        right = data.shape[1] - 1
        while right >= 0 and mx[right] < level:
            right -= 1
        return data[:, left:right]


class RandomCut():
    def __init__(self, window, pad):
        '''
        :param window:
        :param pad: mean, zero
        '''
        self.window = window
        self.pad = pad
        self.valid = True

    def __call__(self, data):
        if data.shape[1] == self.window:
            return data
        if data.shape[1] > self.window:
            shift = np.random.randint(0, data.shape[1] - self.window)
            return data[:, shift:shift + self.window]
        left = np.random.randint(self.window - data.shape[1])
        right = self.window - data.shape[1] - left

        if self.pad == 'zeros':
            return np.pad(data, ((0, 0), (left, right)), 'constant', constant_values=(0,))
        return np.pad(data, ((0, 0), (left, right)), self.pad)


class RandomCutAligned():
    def __init__(self, window, pad, level=0.5):
        self.window = window
        self.pad = pad
        self.level = level
        self.valid = True

    def __call__(self, data):
        if data.shape[1] == self.window:
            return data
        if data.shape[1] > self.window:
            energy = data[:, 0: data.shape[1] - self.window].mean(axis=0)
#            print("E", energy.shape)
            denergy = energy[1:] - energy[:-1]
            if denergy.shape[0] > 0:
                level = self.level * denergy.max()
                index = np.where(denergy >= level)
                index = index[0]
                if index.shape[0] > 0:
                    shift = np.random.choice(index, 1)[0]
                else:
                    shift = 0
            else:
                shift = 0
            data = data[:, shift:shift + self.window]
            return data

        right = self.window - data.shape[1]
        if self.pad == 'zeros':
            data = np.pad(data, ((0, 0), (0, right)), 'constant', constant_values=(0,))
        else:
            data = np.pad(data, ((0, 0), (0, right)), self.pad)
        return data


class RandomCutAudio():
    def __init__(self, window, pad):
        self.window = window
        self.pad = pad
        self.valid = True

    def __call__(self, data):
        if data.shape[0] == self.window:
            return data
        if data.shape[0] > self.window:
            shift = np.random.randint(0, data.shape[0] - self.window)
            data = data[shift:shift + self.window]
            return data

        left = np.random.randint(self.window - data.shape[0])
        right = left + data.shape[0]

        if self.pad == 'mean':
            answer = np.mean(data) * np.ones(self.window)
        else:
            answer = np.zeros(self.window)
        answer[left:right] = data
        return answer


class RandomCutCancat():
    def __init__(self, window, batch, dropout, pad):
        self.window = window
        self.batch = batch
        self.dropout = dropout
        self.pad = pad
        self.valid = True

    def __call__(self, data):
        stacked = []
        for _ in range(self.batch):
            if  data.shape[1] - self.window > 0:
                left = np.random.randint(0, data.shape[1] - self.window)
                right = left + self.window
            else:
                left = 0
                right = self.window

            if self.dropout and data.shape[-1] <= self.window and np.random.uniform(0, 1) < self.dropout:
                if self.pad == 'mean':
                    column = np.mean(data, axis=1).reshape(-1, 1)
                else:
                    column = np.zeros((data.shape[0], 1))
                stacked += [column] * self.window
            else:
                if data.shape[-1] <= self.window:
                    notzero = RandomCut(self.window, self.pad)(data)
                else:
                    notzero = data[:, left:right]
                stacked += [notzero]
        concated_batch = np.hstack(stacked)
        return concated_batch


class UniformCutCancat():
    def __init__(self, window, batch, dropout):
        self.window = window
        self.batch = batch
        self.dropout = dropout
        self.valid = True

    def __call__(self, data):
        if data.shape[-1] <= self.window or self.batch == 1:
            step = 0
        else:
            step = (data.shape[-1] - self.window) // (self.batch - 1)

        stacked = []
        for i in range(self.batch):
            left = i * step
            right = left + self.window
            if self.dropout and data.shape[-1] <= self.window and np.random.randint(self.dropout) == 0:
                stacked += [np.zeros((data.shape[0], self.window))]
            else:
                if data.shape[-1] <= self.window:
                    notzero = RandomShiftWithZeroPad(self.window)(data)
                else:
                    notzero = data[:, left:right]
                stacked += [notzero]
        concated_batch = np.hstack(stacked)
        return concated_batch


class Hanning():
    def __init__(self, batch):
        self.batch = batch
        self.valid = True

    def __call__(self, data):
        window = data.shape[-1] // self.batch
        subkernel = np.hanning(window)
        kernel = np.hstack([subkernel] * self.batch)
        return data * kernel


class Meaning():
    def __init__(self):
        self.valid = True

    def __call__(self, data):
        data[:-1, :-1] = 0.25 * (data[:-1, :-1] + data[1:, :-1] + data[:-1, 1:] + data[1:, 1:])
        return data


class AddNoise():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.valid = True

    def __call__(self, data):
        kernel = np.hanning(2 * data.shape[0])[data.shape[0]:]
        noise = np.random.normal(self.mean, self.std, size=data.shape).T * kernel
        return data + noise.T


class SplitToMiniBatch():
    def __init__(self, window, batch):
        self.window = window
        self.batch = batch
        self.valid = True

    def __call__(self, data):
        if data.shape[-1] <= self.window:
            concated_batch = np.array([data] * self.batch)
            return concated_batch

        step = (data.shape[-1] - self.window) // (self.batch - 1)
        batch_data = []
        for i in range(self.batch):
            left = i * step
            right = left + self.window
            batch_data.append(data[:, :, left:right])
        concated_batch = np.array(batch_data)
        return concated_batch


class AddChannelDimension():
    def __init__(self):
        self.valid = True

    def __call__(self, data):
        return data.reshape((1, data.shape[0], data.shape[1])).astype('float32')


class AddChannelDimensionAudio():
    def __init__(self):
        self.valid = True

    def __call__(self, data):
        return data.reshape((1, data.shape[0])).astype('float32')


class RGB():
    def __init__(self, channel, inplace, normalize):
        self.channel = channel
        self.inplace = bool(inplace)
        self.normalize = normalize
        self.valid = True

    def __call__(self, data):
        data_channel = data[self.channel]
        if self.normalize:
            data_channel = plt.Normalize()(data_channel)
        rgba = plt.cm.jet(data_channel)
        rgba = np.swapaxes(rgba, 0, 2)
        rgba = np.swapaxes(rgba, 1, 2)
        rgb = rgba[0:3]
        if self.inplace:
            return rgb
        return np.concatenate([data, rgb])


class MovingKernel():
    def __init__(self, windows, hop, types=['max', 'min', 'mean']):
        self.types = types
        self.windows = windows
        self.hop = hop
        self.valid = True

    def moving(self, func, width, hop, data):
        answer = []
        for i in range(0, data.shape[0], hop):
            answer.append(func(data[i: i + width]))
        return np.array(answer)

    def __call__(self, data):
        width = data.shape[0] - (self.windows - 1) * self.hop
        answer = []
        if width <= 0:
            raise Exception("MovingKernel has bad windows and hop parameters: width = " + str(width))
        for ch in range(len(self.types)):
            answer.append(self.moving(getattr(np, self.types[ch]), width, self.hop, data))
        return np.vstack(answer)


class RandomHorizontalFlip():
    def __init__(self, prob):
        self.prob = prob
        self.valid = True

    def __call__(self, data):
        if np.random.uniform(0, 1) < self.prob:
            return data[:, ::-1]
        return data


class Aug():
    def __init__(self, freq_mask_num=2, time_mask_num=0, freq_mask_prob=0.15, time_mask_prob=0.3, pad='zeros', validation=True):
        self.freq_mask_num = freq_mask_num
        self.time_mask_num = time_mask_num
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob
        self.pad = pad
        self.valid = validation

    def __call__(self, data):
        if self.pad == 'zeros':
            pvalue = 0
        if self.pad == 'mean':
            pvalue = data.mean()
        for _ in range(self.time_mask_num):
            mask_width = int(np.random.uniform(0.0, self.time_mask_prob) * data.shape[1])
            mask_start = int(np.random.uniform(low=0.0, high=data.shape[1] - mask_width))
            data[:, mask_start:mask_start + mask_width] = pvalue

        for _ in range(self.freq_mask_num):
            mask_height = int(np.random.uniform(0.0, self.freq_mask_prob) * data.shape[0])
            mask_start = int(np.random.uniform(low=0.0, high=data.shape[0] - mask_height))
            data[mask_start:mask_start + mask_height, :] = pvalue
        return data


class ToTensor():
    def __init__(self):
        self.valid = True

    def __call__(self, data):
        return torch.from_numpy(data.astype('float32'))
