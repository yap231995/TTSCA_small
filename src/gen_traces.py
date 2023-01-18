import random

import numpy as np

from tqdm import tqdm
import argparse
from src.config2 import Config
from src.utils import str2bool


class TraceGenerator(object):
    # n_features, n_traces, var_noise, k
    def __init__(self, config):
        self.config = config
        self.n_features = config.n_features
        self.X_profiling_traces = config.X_profiling_traces
        self.X_attack_traces = config.X_attack_traces
        self.var_noise = config.var_noise
        self.k = config.k
        self.leakage_distance = config.leakage_distance #currently only for order 1.
        self.leakage_shift = config.leakage_shift #currently only for order 1.
        self.accidental_leakage = config.accidental_leakage
        self.accidental_leakage_order = config.accidental_leakage_order
        self.SBox = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]



    def HW(self, val):
        # compute the HW of val
        h = bin(int(val)).count("1")
        return h

    # Leakage = SBox(p xor k) + noise at sample 10
    # Label = SBox(p xor k)

    def gen_trace(self):
        trace = np.random.randint(256, size=self.n_features)
        p = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        # trace[10] = self.HW(label)
        trace[10] = label
        trace = trace + np.random.normal(0, self.var_noise, self.n_features)
        return trace, label, p

    # Leakage = SBox(p xor k) xor m + noise at sample 10
    # mask at sample 5
    # Label = SBox(p xor k)

    def gen_trace_mask_order1(self):
        trace = np.random.randint(256, size=self.n_features)
        p = np.random.randint(256)
        m = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        # trace[10] = self.HW(label ^ m)
        # trace[5] = self.HW(m)
        trace[10] = label ^ m
        trace[5] = m
        trace = trace + np.random.normal(0, self.var_noise, self.n_features)
        return trace, label, p

    def gen_trace_mask_order1_leakage_not_in_patch(self):
        trace = np.random.randint(256, size=self.n_features)
        p = np.random.randint(256)
        m = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        # trace[10] = self.HW(label ^ m)
        # trace[5] = self.HW(m)
        trace[10] = label ^ m
        trace[1] = m
        trace = trace + np.random.normal(0, self.var_noise, self.n_features)
        return trace, label, p


    def gen_trace_mask_order1_leakage_distance(self, shift = 0, distance = 5):
        trace = np.random.randint(256, size=self.n_features)
        p = np.random.randint(256)
        m = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        index = 5+shift
        trace[index + distance] = label ^ m
        trace[index] = m
        trace = trace + np.random.normal(0, self.var_noise, self.n_features)
        return trace, label, p



    # Leakage = SBox(p xor k) xor m + noise at sample 10
    # mask at sample 5 is removed
    # Label = SBox(p xor k) + noise

    def gen_trace_mask_order1_nomaskleak(self):
        trace = np.random.randint(256, size=self.n_features)
        p = np.random.randint(256)
        m = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        trace[10] = label ^ m
        # trace[5]= m
        trace = trace + np.random.normal(0, self.var_noise, self.n_features)
        return trace, label, p

    # Leakage = SBox(p xor k) xor m1 xor m2 + noise at sample 10
    # m1 and m2 at sample 5 and 15
    # Label = SBox(p xor k)

    def gen_trace_mask_order2(self):
        trace = np.random.randint(256, size=self.n_features)
        p = np.random.randint(256)
        m1 = np.random.randint(256)
        m2 = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        # trace[10] = self.HW(label ^ m1 ^ m2)
        # trace[5] = self.HW(m1)
        # trace[8] = self.HW(m2)
        trace[10] = label ^ m1 ^ m2
        trace[5] = m1
        trace[8] = m2
        trace = trace + np.random.normal(0, self.var_noise, self.n_features)
        return trace, label, p

    # Leakage = SBox(p xor k) xor m1 xor m2 xor m3 + noise at sample 10
    # m1, m2 and m3 at sample 5, 15 and 18
    # Label = SBox(p xor k)

    def gen_trace_mask_order3(self):
        trace = np.random.randint(256, size=self.n_features)
        p = np.random.randint(256)
        m1 = np.random.randint(256)
        m2 = np.random.randint(256)
        m3 = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        trace[10] = label ^ m1 ^ m2 ^ m3
        trace[5] = m1
        trace[8] = m2
        trace[12] = m3
        # trace[10] = self.HW(label ^ m1 ^ m2 ^ m3)
        # trace[5] = self.HW(m1)
        # trace[8] = self.HW(m2)
        # trace[12] = self.HW(m3)
        trace = trace + np.random.normal(0, self.var_noise, self.n_features)
        return trace, label, p

    def gen_trace_mask_order3_leakage_not_in_patch(self):
        trace = np.random.randint(256, size=self.n_features)
        p = np.random.randint(256)
        m1 = np.random.randint(256)
        m2 = np.random.randint(256)
        m3 = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        trace[10] = label ^ m1 ^ m2 ^ m3
        trace[5] = m1
        trace[8] = m2
        trace[15] = m3
        # trace[10] = self.HW(label ^ m1 ^ m2 ^ m3)
        # trace[5] = self.HW(m1)
        # trace[8] = self.HW(m2)
        # trace[12] = self.HW(m3)
        trace = trace + np.random.normal(0, self.var_noise, self.n_features)
        return trace, label, p


    def gen_trace_accidental_leakage(self, curr_order, leakage_order):
        trace = np.random.randint(256, size=self.n_features)
        p = np.random.randint(256)
        m1 = np.random.randint(256)
        m2 = np.random.randint(256)
        m3 = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        if curr_order == 1:
            trace[10] = label ^ m1
            trace[5] = m1
            trace[15] = label
        if curr_order ==2:
            trace[10] = label ^ m1 ^ m2
            trace[5] = m1
            trace[8] = m2
            if leakage_order == 1:
                trace[15] = m1 ^ m2 #leakage
            if leakage_order == 0:
                trace[15] = label #leakage
        if curr_order == 3:
            trace[10] = label ^ m1 ^ m2 ^ m3
            trace[5] = m1
            trace[8] = m2
            trace[12] = m3

            if leakage_order == 0:
                trace[15] = label
            if leakage_order == 1:
                trace[15] = m1 ^ m2 ^ m3
            if leakage_order == 2:
                if self.config.accidental_leakage_order_3_type == 'm1m2':
                    print("m1m2")
                    trace[15] = m1 ^ m2
                elif self.config.accidental_leakage_order_3_type == 'm2m3':
                    print("m2m3 shouldnt be here")
                    trace[15] = m2 ^ m3
                elif self.config.accidental_leakage_order_3_type == 'm1m3':
                    print("m1m3")
                    trace[15] = m1 ^ m3
        trace = trace + np.random.normal(0, self.var_noise, self.n_features)
        return trace, label, p

    def gen_trace_accidental_leakage_same_patch(self, curr_order, leakage_order):
        trace = np.random.randint(256, size=self.n_features)
        p = np.random.randint(256)
        m1 = np.random.randint(256)
        m2 = np.random.randint(256)
        m3 = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        if curr_order == 1:
            trace[10] = label ^ m1
            trace[5] = m1
            trace[12] = label
        if curr_order ==2:
            trace[10] = label ^ m1 ^ m2
            trace[5] = m1
            trace[8] = m2
            if leakage_order == 1:
                trace[12] = m1 ^ m2 #leakage
            if leakage_order == 0:
                trace[12] = label #leakage
        if curr_order == 3:
            trace[10] = label ^ m1 ^ m2 ^ m3
            trace[5] = m1
            trace[8] = m2
            trace[12] = m3

            if leakage_order == 0:
                trace[6] = label
            if leakage_order == 1:
                trace[6] = m1 ^ m2 ^ m3
            if leakage_order == 2:
                if self.config.accidental_leakage_order_3_type == 'm1m2':
                    print("m1m2 same patch")
                    trace[6] = m1 ^ m2
                elif self.config.accidental_leakage_order_3_type == 'm2m3':
                    print("m2m3 same patch")
                    trace[6] = m2 ^ m3
                elif self.config.accidental_leakage_order_3_type == 'm1m3':
                    print("m1m3 same patch")
                    trace[6] = m1 ^ m3
        trace = trace + np.random.normal(0, self.var_noise, self.n_features)
        return trace, label, p

    def generate_traces(self, n_traces, order = 0, nomaskleak = True):
        ### nomaskleak: True/False/None

        traces = np.zeros((n_traces, self.n_features))
        labels = np.zeros((n_traces, 1))
        plaintexts = np.zeros((n_traces, 1))
        for i in tqdm(range(n_traces)):
            trace=None
            label=None
            p = None
            if self.accidental_leakage == False:
                if order == 0:
                    trace, label, p = self.gen_trace()
                elif order == 1 and nomaskleak == False and self.leakage_distance == 5 and self.leakage_shift == 0 and self.config.leakage_not_in_patch == False: ##This is because of the randomness is different if we do not use this.
                    print("in patch")
                    trace, label, p  = self.gen_trace_mask_order1()
                elif order == 1 and nomaskleak == False and self.leakage_distance == 5 and self.leakage_shift == 0 and self.config.leakage_not_in_patch == True: ##This is because of the randomness is different if we do not use this.
                    print("not_in_patch")
                    trace, label, p  = self.gen_trace_mask_order1_leakage_not_in_patch()
                elif order == 1 and nomaskleak == False and self.leakage_shift != 0:
                    single_shift = random.randint(0, self.leakage_shift)
                    trace, label, p = self.gen_trace_mask_order1_leakage_distance(distance=self.leakage_distance, shift = single_shift)

                elif order == 1 and nomaskleak == True:
                    trace, label, p  = self.gen_trace_mask_order1_nomaskleak()
                elif order == 2:
                    trace, label, p  = self.gen_trace_mask_order2()
                elif order == 3 and self.config.leakage_not_in_patch == False:
                    print("gen_trace_mask_order3")
                    trace, label, p  = self.gen_trace_mask_order3()
                elif order == 3 and self.config.leakage_not_in_patch == True:
                    print("gen_trace_mask_order3_leakage_not_in_patch")
                    trace, label, p = self.gen_trace_mask_order3_leakage_not_in_patch()


            elif self.accidental_leakage == True:
                # print("HELLO? A")
                curr_order = order
                leakage_order = self.accidental_leakage_order
                if self.config.all_leakage_in_same_patch == False:
                    print("gen_trace_accidental_leakage all in NOT! same patch")
                    trace, label, p = self.gen_trace_accidental_leakage(curr_order, leakage_order)
                elif self.config.all_leakage_in_same_patch==True:
                    print("gen_trace_accidental_leakage all in same patch")
                    trace, label, p = self.gen_trace_accidental_leakage_same_patch(curr_order, leakage_order)
            traces[i, :] = trace
            labels[i, :] = label
            plaintexts[i,:] = p
        traces = traces.astype(np.int8)
        labels = labels.astype(np.int64)
        plaintexts = plaintexts.astype(np.uint8)
        return traces, labels, plaintexts

    def load_traces(self, order = 0, nomaskleak = False, distance = 5):
        X_profiling_traintot, Y_profiling_traintot, plaintexts = self.generate_traces(self.X_profiling_traces, order, nomaskleak)
        X_profiling_test, _, plaintexts = self.generate_traces(self.X_attack_traces, order, nomaskleak)
        Y_profiling_traintot = np.squeeze(Y_profiling_traintot)
        real_key = self.k
        #create all posible labels over all keys.
        Y_profiling_test = []
        pla = plaintexts.tolist()
        for plaintext in pla:
            possilabel = []
            for possiKey in range(0,256):
                x = self.SBox[int(plaintext[0])^possiKey]
                possilabel.append(x)
            Y_profiling_test.append(possilabel)
        Y_profiling_test = np.asarray(Y_profiling_test)
        return X_profiling_traintot, Y_profiling_traintot, X_profiling_test, Y_profiling_test, real_key

    def gen_long_single_trace_order_0(self):
        trace = np.random.randint(256, size=self.long_trace_n_features)
        p = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        if self.config.mode == "HW":
            m_2 = self.HW(label)
        else:
            m_2 = label

        if self.m_1_length == 1:
            trace[150] = m_2
        else:
            for i in range(self.m_1_length):
                trace[i] = m_2
                if i < self.m_1_length/2:
                    trace[i] += np.random.normal(0, (0.1 *(self.m_1_length/2 - i)), 1)
                elif i >=self.m_1_length/2:
                    trace[i] += np.random.normal(0, (0.1 * (-1*(self.m_1_length/2-1 - i))), 1)


        trace = trace + np.random.normal(0, self.var_noise, self.long_trace_n_features)
        return trace, label, p, None, m_2

    def gen_long_single_trace_order_1(self):
        trace = np.random.randint(256, size=self.long_trace_n_features)
        p = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        if self.config.mode == "HW":
            m_1 = np.random.randint(256)
            m_2 = label ^ m_1
            m_1 = self.HW(m_1)
            m_2 = self.HW(m_2)
        else:
            m_1 = np.random.randint(256)
            m_2 = label ^ m_1

        if self.m_1_length == 1:
            trace[150] = m_1
        else:
            for i in range(self.m_1_length):
                trace[i] = m_1
                if i < self.m_1_length/2:
                    trace[i] += np.random.normal(0, (0.1 *(self.m_1_length/2 - i)), 1)
                elif i >=self.m_1_length/2:
                    trace[i] += np.random.normal(0, (0.1 * (-1*(self.m_1_length/2-1 - i))), 1)
        if self.m_2_length == 1:
            trace[510] = m_2
        else:
            for index,i in enumerate(range(self.long_trace_n_features-1,self.long_trace_n_features - self.m_2_length-1, -1)):
                trace[i] = m_2 + np.random.normal(0, (0.1* (self.m_2_length-1-index)), 1)
        trace = trace + np.random.normal(0, self.var_noise, self.long_trace_n_features)
        return trace, label, p, m_1, m_2


    def gen_long_single_trace_order_1_overlap(self):
        trace = np.random.randint(256, size=self.long_trace_n_features)
        p = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        if self.config.mode == "HW":
            m_1 = np.random.randint(256)
            m_2 = label ^ m_1
            m_1 = self.HW(m_1)
            m_2 = self.HW(m_2)
        else:
            m_1 = np.random.randint(256)
            m_2 = label ^ m_1
        assert self.m_1_length == self.m_2_length, "m_1_length and m_2_length are not equal"
        for i in range(75, 75+self.m_1_length):
            trace[i] = m_1 + m_2
        trace = trace + np.random.normal(0, self.var_noise, self.long_trace_n_features)
        return trace, label, p, m_1, m_2

    def gen_long_single_trace_order_1_side_by_side(self):
        trace = np.random.randint(256, size=self.long_trace_n_features)
        p = np.random.randint(256)
        label = self.SBox[p ^ self.k]
        if self.config.mode == "HW":
            m_1 = np.random.randint(256)
            m_2 = label ^ m_1
            m_1 = self.HW(m_1)
            m_2 = self.HW(m_2)
        else:
            m_1 = np.random.randint(256)
            m_2 = label ^ m_1
        for i in range(75, 75+self.m_1_length):
            trace[i] = m_1
        for j in range(75+self.m_1_length, 75+self.m_1_length + self.m_2_length):
            trace[j] = m_2
        trace = trace + np.random.normal(0, self.var_noise, self.long_trace_n_features)
        return trace, label, p, m_1, m_2

    def generate_long_traces(self,n_traces):
        traces = np.zeros((n_traces, self.long_trace_n_features))
        labels = np.zeros((n_traces, 1))
        plaintexts = np.zeros((n_traces, 1))
        m_1s = np.zeros((n_traces, 1))
        m_2s = np.zeros((n_traces, 1))
        assert not (self.config.long_side_by_side == True and self.config.long_overlap == True), "Both long_side_by_side and long_overlap are True, choose either one is True or both False"
        for i in tqdm(range(n_traces)):
            if self.config.long_masking_order == 0:
                trace, label, p, m_1, m_2 = self.gen_long_single_trace_order_0()
            if self.config.long_masking_order == 1 and self.config.long_overlap == False and self.config.long_side_by_side == False:
                trace, label, p, m_1, m_2 = self.gen_long_single_trace_order_1()
            elif self.config.long_masking_order == 1 and self.config.long_overlap == True and self.config.long_side_by_side == False:
                trace, label, p, m_1, m_2 = self.gen_long_single_trace_order_1_overlap()
            elif self.config.long_masking_order == 1 and self.config.long_overlap == False and self.config.long_side_by_side == True:
                trace, label, p, m_1, m_2 = self.gen_long_single_trace_order_1_side_by_side()
            traces[i, :] = trace
            labels[i, :] = label
            plaintexts[i, :] = p
            if m_1 != None:
                m_1s[i, :] = m_1
            else:
                m_1s = None
            m_2s[i, :] = m_2
        traces = traces.astype(np.int8)
        labels = labels.astype(np.int64)
        plaintexts = plaintexts.astype(np.uint8)
        return traces, labels, plaintexts, m_1s,m_2s


    def load_long_traces(self):
        X_profiling_traintot, Y_profiling_traintot, plaintexts,_,_ = self.generate_long_traces(self.X_profiling_traces)
        X_profiling_test, _, plaintexts, _, _= self.generate_long_traces(self.X_attack_traces)
        Y_profiling_traintot = np.squeeze(Y_profiling_traintot)
        real_key = self.k
        # create all posible labels over all keys.
        Y_profiling_test = []
        pla = plaintexts.tolist()
        for plaintext in pla:
            possilabel = []
            for possiKey in range(0, 256):
                x = self.SBox[int(plaintext[0]) ^ possiKey]
                possilabel.append(x)
            Y_profiling_test.append(possilabel)
        Y_profiling_test = np.asarray(Y_profiling_test)
        return X_profiling_traintot, Y_profiling_traintot, X_profiling_test, Y_profiling_test, real_key


if __name__ == '__main__':
    config = Config(path="../config/")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=config.general.seed, type=int, choices=[i for i in range(100)])
    parser.add_argument("--n_features", default=config.gen_mask_traces.n_features, type=int)
    parser.add_argument("--X_profiling_traces", default=config.gen_mask_traces.X_profiling_traces, type=int)
    parser.add_argument("--X_attack_traces", default=config.gen_mask_traces.X_attack_traces,type=int)
    parser.add_argument("--var_noise", default=config.gen_mask_traces.var_noise)
    parser.add_argument("--k", default=config.gen_mask_traces.k, type = int, choices=[i for i in range(256)])
    parser.add_argument("--masking_order", default=config.gen_mask_traces.masking_order, type = int, choices=[i for i in range(4)])
    parser.add_argument("--nomaskleak", default=config.gen_mask_traces.nomaskleak, type=str2bool, choices=[True, False])

    config = parser.parse_args()
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    Generator = TraceGenerator(config)
    X_profiling_traintot, Y_profiling_traintot, X_profiling_test, Y_profiling_test, real_key = Generator.load_traces(config.masking_order)
    # print(X_profiling_traintot)
    # print(Y_profiling_traintot)
    # print(X_profiling_test)
    # print(Y_profiling_test)
    # print(real_key)

    # def gen_trace_allzeros(self):
    #     trace = np.zeros(shape= self.n_features)
    #     p = np.random.randint(256)
    #     label = self.SBox[p ^ self.k]
    #     trace[10] = label
    #     trace = trace
    #     return trace, label, p
    #
    #
    # def gen_trace_mask_order1_allzeros(self):
    #     trace = np.zeros(shape= self.n_features)
    #     p = np.random.randint(256)
    #     m = np.random.randint(256)
    #     label = self.SBox[p ^ self.k]
    #     trace[10] = label ^ m
    #     trace[5] = m
    #     trace = trace
    #     return trace, label, p
    #
    # def gen_trace_mask_order1_nomaskleak_allzeros(self):
    #     trace = np.zeros(shape=self.n_features)
    #     p = np.random.randint(256)
    #     m = np.random.randint(256)
    #     label = self.SBox[p ^ self.k]
    #     trace[10] = label ^ m
    #     # trace[5]= m
    #     trace = trace
    #     return trace, label, p
    #
    # # Leakage = SBox(p xor k) xor m1 xor m2 + noise at sample 10
    # # m1 and m2 at sample 5 and 15
    # # Label = SBox(p xor k)
    #
    # def gen_trace_mask_order2_allzeros(self):
    #     trace = np.zeros(shape= self.n_features)
    #     p = np.random.randint(256)
    #     m1 = np.random.randint(256)
    #     m2 = np.random.randint(256)
    #     label = self.SBox[p ^ self.k]
    #     trace[10] = label ^ m1 ^ m2
    #     trace[10] = label ^ m1 ^ m2
    #     trace[5] = m1
    #     trace[15] = m2
    #     trace = trace
    #     return trace, label, p
    #
    # # Leakage = SBox(p xor k) xor m1 xor m2 xor m3 + noise at sample 10
    # # m1, m2 and m3 at sample 5, 15 and 18
    # # Label = SBox(p xor k)
    #
    # def gen_trace_mask_order3_allzeros(self):
    #     trace = np.zeros(shape=self.n_features)
    #     p = np.random.randint(256)
    #     m1 = np.random.randint(256)
    #     m2 = np.random.randint(256)
    #     m3 = np.random.randint(256)
    #     label = self.SBox[p ^ self.k]
    #     trace[10] = label ^ m1 ^ m2 ^ m3
    #     trace[5] = m1
    #     trace[15] = m2
    #     trace[18] = m3
    #     trace = trace
    #     return trace, label, p
    #
    # def generate_traces_allzeros(self, order = 0, nomaskleak = True):
    #     ### nomaskleak: True/False/None
    #     traces = np.zeros((self.n_traces, self.n_features))
    #     labels = np.zeros((self.n_traces, 1))
    #     plaintexts = np.zeros((self.n_traces, 1))
    #
    #     for i in tqdm(range(self.n_traces)):
    #         p = None
    #         trace=None
    #         label=None
    #         if order == 0:
    #             trace, label, p  = self.gen_trace_allzeros()
    #         elif order == 1 and nomaskleak == True:
    #             trace, label, p  = self.gen_trace_mask_order1_allzeros()
    #         elif order == 1 and nomaskleak == False:
    #             trace, label , p = self.gen_trace_mask_order1_nomaskleak_allzeros()
    #         elif order == 2:
    #             trace, label , p = self.gen_trace_mask_order2_allzeros()
    #         elif order == 3:
    #             trace, label , p = self.gen_trace_mask_order3_allzeros()
    #
    #         traces[i, :] = trace
    #         labels[i, :] = label
    #         plaintexts[i, :] = p
    #     return traces, labels, plaintexts

