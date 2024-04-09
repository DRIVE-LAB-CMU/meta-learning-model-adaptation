# from asyncio.windows_events import NULL
import cv2
import numpy as np
import glob
import argparse
import tqdm
import os

RUN_NO = 5
GP_LEN = 305
PATH = 'RUN_ONLINE_'+str(RUN_NO)+'_'+str(GP_LEN)+'/Video/'
argparser = argparse.ArgumentParser()
argparser.add_argument(
        '-n', '--run_no',
        metavar='P',
        default=-1,
        type=int,
        help='Run no')
args = argparser.parse_args()

if args.run_no != -1 :
    RUN_NO = args.run_no


INPUT_FOLDER = PATH
OUTPUT_FILE = 'RUN_ONLINE_'+str(RUN_NO)+'_'+str(GP_LEN)+'/video.mp4'
img_array = []
# print(INPUT_FOLDER+'/*')
file_list = glob.glob(INPUT_FOLDER+'/*')
n_files = len(file_list)
file_list.sort()
file_list = []
for i in range(n_files) :
    file_list.append(INPUT_FOLDER+'/frame'+str(i)+'.png')
# print(file_list)
for filename in file_list:
    # print(filename)
    img = cv2.imread(filename)
    if img is not None:
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


out = cv2.VideoWriter(OUTPUT_FILE,cv2.VideoWriter_fourcc(*'mp4v'), 25, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()