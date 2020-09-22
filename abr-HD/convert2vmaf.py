import os
import numpy as np

DIR = './norway'
SUMMARY_DIR = DIR + '_vmaf/'
VIDEO_QUALITY_FILE = './envivio/vmaf/video_'

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_LEVELS = len(VIDEO_BIT_RATE)
# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

video_vmaf = {}  # in bytes
for bitrate in range(BITRATE_LEVELS):
    video_vmaf[bitrate] = []
    with open(VIDEO_QUALITY_FILE + str(bitrate)) as f:
        for line in f:
            video_vmaf[bitrate].append(float(line.split()[0]))
            
for filename in os.listdir(DIR + '/'):
    _filename = DIR + '/' + filename
    fw = open(SUMMARY_DIR + filename, 'w')
    f = open(_filename, 'r')
    for _chunk_idx, line in enumerate(f):
        sp = line.split()
        if len(sp) > 0:
            _timestamp = float(sp[0])
            _bitrate = int(sp[1])
            _buffer = float(sp[2])
            _rebuffer = float(sp[3])
            _download = int(sp[4])
            _timespan = float(sp[5])
            _qoe = float(sp[6])
            _chunk_br = np.argmin(np.abs(_bitrate - np.array(VIDEO_BIT_RATE)))
            _vmaf = video_vmaf[_chunk_br][_chunk_idx]
            fw.write(line.replace('\n', '') + '\t' + str(_vmaf) + '\n')
        else:
            fw.write(line)

    f.close()
    fw.close()