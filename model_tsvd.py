#!/usr/bin/env python

import numpy as np

TC = np.genfromtxt('Gordon.csv', delimiter=',')
lag = 5 #indicates how many autoregressive terms to remove
perc = [0.7, 0.3]

def rm_ac(TC):
    [samples, N] = TC.shape #360x333
    y_pred = np.zeros((samples-lag, N)) #355x333
    R = np.zeros((N, 1)) #initialize residuals vector 360x1

    for roi in range(0,N):
        y = (TC[lag:,roi]) #vector of length 355
        x = np.zeros((samples-lag,lag)) #initialize 355x5 matrix
        for j in range(0,lag):
            x[j+1:,j] = TC[lag:samples-j-1,roi]
        pinv_x = np.linalg.pinv(x)
        coefs = np.inner(pinv_x, y)
        y_pred[:,roi] = np.inner(x, coefs)
        R[roi] = np.corrcoef(y, y_pred[:,roi])[1,0]

    TC_no_AC_LS = np.subtract(TC[lag:samples,:], y_pred)
    return TC_no_AC_LS
TC_no_AC_LS = rm_ac(TC)

def run_svd(y_in, y_out, A, rois, test_out, i, j, R, inc_frames):
    [U, S, V] = np.linalg.svd(A)
    x_inv = np.zeros((rois-1, inc_frames))
    for l in range(len(y_in)):
        x_inv = x_inv + np.outer(V[:,l], U[:,l].T)*(1/S[l])
        local_c = np.inner(x_inv, y_in)
        yp = np.inner(test_out, local_c)
        R[i,j,l] = np.corrcoef(yp, y_out)[0][1]

def model_tsvd(signal):
    rep = 10
    [frames, rois] = signal.shape
    mask = range(1,rois+1)
    inc_frames = round(frames*perc[0])
    max_SV = min([inc_frames, rois-1])
    R = np.zeros((rep, rois, max_SV))
    for i in range(rep):
        ix = np.random.permutation(frames)
        ix_in = ix[0:inc_frames] #frames for training set?
        ix_out = ix[(inc_frames-len(ix)):len(ix)] #frames for testing set?
        TC_in = signal[ix_in, :]
        TC_out = signal[ix_out, :]
        roi_list = []
        for j in range(rois):
            y_in = TC_in[:,j]
            y_out = TC_out[:,j]
            A = np.delete(TC_in, j, 1)
            run_svd(y_in, y_out, A, rois, np.delete(TC_out, j, 1), i, j, R, inc_frames)
        print("SVD decomposition, %d out of %d" % (i, rep))
    mr = np.mean(R, axis=0)
    SV = np.argmax(mr, axis=1)
    m = np.amax(mr, axis=1)
    return [m, SV]
[m, SV] = model_tsvd(TC_no_AC_LS)
print m
print SV




    


