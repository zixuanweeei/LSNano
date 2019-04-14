# coding: utf-8

import numpy as np

def segment(signal, meanOpenCurr, eventThreshold, sdOpenCurr, currentThreshold=None):
    if currentThreshold is None:
        currentThreshold = abs(meanOpenCurr) - eventThreshold*abs(sdOpenCurr)
    
    # Make sure that the values of the signal are all positive.
    # This can be also done with presetted offset and scale when reading the 
    # original data.
    signal = np.abs(signal)

    # Mark all the points that not great than `currentThreshold` as the doubt points
    # with 1s and the others with 0s
    marks = signal <= currentThreshold
    for idx, mark in enumerate(marks):
        print("Escape: {0:.2f}%({1}) points passed ...".format(idx/marks.shape[0]*100, idx), end="\r")        
        if mark:
            continue
        else:
            break
    
    isinBlockage = False
    eventStartStop = []
    while idx < marks.shape[0]:
        if not marks[idx] and not isinBlockage:
            idx += 1
            print("Open: {0:.2f}%({1}) points passed ...".format(idx/marks.shape[0]*100, idx), end="\r")
            continue
        
        if marks[idx] and not isinBlockage:
            isinBlockage = True
            eventStart = idx
            while isinBlockage and marks[idx]:
                idx += 1
                print("Blockage: {0:.2f}%({1}) points passed ...".format(idx/marks.shape[0]*100, idx), end="\r")
            isinBlockage = False
            eventStop = idx
            eventStartStop.append([eventStart, eventStop])

    return eventStartStop

def segment2(signal, meanOpenCurr, eventThreshold, sdOpenCurr, currentThreshold=None):
    """ Segment the events using threshold.
    Parameters:
    -----------
    signal : `array` or `list`
        Raw current signal
    meanOpenCurrent : `float`
        Open current value
    eventThreshold : `float`
        Determine the threshold below which the signal can be considered as events
    sdOpenCurr : `float`
        Standard deviation of the open current
    currentThreshold : `float`
        Direct way to set threshold of events
    
    Returns:
    --------
    events_start_stop : `array`
        An array containing the events beginning point index and that of the events end. Each event is recorded in row.
    """
    if currentThreshold is None:
        currentThreshold = abs(meanOpenCurr) - eventThreshold*abs(sdOpenCurr)
    
    # Make sure that the values of the signal are all positive.
    # This can be also done with presetted offset and scale when reading the 
    # original data.
    signal = np.abs(signal)

    # Mark all the points that not great than `currentThreshold` as the doubt points
    # with 1s and the others with 0s
    marks = signal <= currentThreshold
    for idx, mark in enumerate(marks):
        if mark:
            continue
        else:
            break
    escape = idx
    marks = marks[idx:]
    for idx, mark in enumerate(np.flip(marks, 0)):
        if mark:
            continue
        else:
            break
    if idx > 0:
        marks = marks[:-idx]

    marks = marks.astype(int)
    marksChange = np.abs(np.diff(marks))
    startStop = np.where(marksChange > 0.0)
    startStop = startStop[0] + escape
    startStop = startStop.reshape((-1, 2))
    # width = np.diff(startStop, axis=-1)
    # dropIdx = np.where(width < 10.)[0]
    # startStop = np.delete(startStop, (dropIdx, ), 0)
            
    return startStop

def eventsHistogram(startStop, bins="auto", logarothmic=False):
    eventsResTime = np.diff(startStop, axis=-1)
    if logarothmic:
        hist, bin_edges = np.histogram(np.log2(eventsResTime), bins=bins)
    else:
        hist, bin_edges = np.histogram(eventsResTime, bins=bins)
    
    return hist, bin_edges
    

if __name__ == "__main__":
    import os
    from neo import io
    import pandas as pd
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    scale = -0.0135
    offset = 1396.3
    dataPath = '../data/deep learning/similar signal/'
    currentSignal = loadmat(os.path.join(dataPath, "PA1.mat"))["PA1"]
    currentSignal = currentSignal.squeeze()
    # abfFile = os.path.join(dataPath, '15o15012.abf')

    # reader = io.AxonIO(filename=abfFile)
    # bl = reader.read_block()
    # currentSignal = bl.segments[0].analogsignals[0].squeeze().magnitude
    # currentSignal = (offset + currentSignal)*scale
    fig_signal, ax_signal = plt.subplots()
    ax_signal.plot(currentSignal[:100000])
    fig_signal.show()
    startStop = segment2(currentSignal, 48.5, 2, 5, 40)
    startStop = pd.DataFrame(startStop, columns=['start', 'stop'])
    startStop.to_csv('./events_n.csv')
    hist, bin_edges = eventsHistogram(startStop, logarothmic=True)
    x = []
    for idx in range(bin_edges.shape[0] - 1):
        x.append((bin_edges[idx] + bin_edges[idx + 1])/2)
    fig_hist, ax_hist = plt.subplots()
    ax_hist.bar(x, hist, bin_edges[1] - bin_edges[0])
    fig_hist.show()
