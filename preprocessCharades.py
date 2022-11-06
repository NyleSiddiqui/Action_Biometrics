import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
# import torch
import cv2
# from torch import nn, einsum
# from einops import rearrange, repeat
import timeit
# from decord import VideoReader, cpu

def splitActions(df):
    df.drop(['quality', 'relevance', 'script', 'objects', 'descriptions', 'verified'], axis=1, inplace=True)
    df = df.dropna(subset = ['actions'])
    new_df_rows = []
    for count, row in enumerate(df.to_numpy()):
        try:
            ast = row[3].split(';')
        except AttributeError:
            ast = row[3]
        for actions in ast:
            action, start, end = actions.split(' ')
            new_df_rows.append([row[0], row[1], row[2], action[1:], start, end])
    new_df = pd.DataFrame(new_df_rows, columns=['id', 'subject', 'scene', 'action', 'start', 'end'])
    new_df.reset_index()
    new_df.to_csv("CharadesEgo_test.csv")


def indexNTU():
    path = "/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_+rgb"
    df_rows = []
    for row in os.listdir(path):
        if row[16:20] in ['A050', 'A051', 'A052', 'A053', 'A054', 'A055', 'A056', 'A057', 'A058',
                          'A059', 'A060', 'A106', 'A107', 'A108', 'A109', 'A110', 'A111', 'A112',
                          'A113', 'A114', 'A115', 'A116', 'A117', 'A118', 'A119', 'A120']:
            continue
        else:
            s_num, cam_id, sub_id, rep_num, act_id = row[0:4], row[4:8], row[8:12], row[12:16], row[16:20]
            if sub_id in range(1, 71):
                df_rows.append([row, s_num[1:], cam_id[1:], sub_id[1:], rep_num[1:], act_id[1:]])
    df = pd.DataFrame(df_rows, columns=['video_id', 'setup', 'camera', 'subject', 'repetition', 'action'])
    df.reset_index()
    df.to_csv("NTU_map.csv")

def indexPK():
    path = "C://Users/ny525072/Downloads/label"
    trdf_rows = []
    ttdf_rows = []
    for count, text_file in enumerate(os.listdir(path)):
        sub_id = int(text_file[4:6])   # + 66
        for line in open(os.path.join(path, text_file), 'r').readlines():
            if len(line.split(',')) == 4:
                act_id, start_frame, end_frame, confidence = line.split(',')
                confidence = confidence[0]
            else:
                act_id, start_frame, end_frame = line.split(',')
                confidence = 2
            if sub_id in range(1, 11):
                trdf_rows.append([f"{text_file[:-4]}", sub_id, act_id, start_frame, end_frame, confidence])
            else:
                ttdf_rows.append([f"{text_file[:-4]}", sub_id, act_id, start_frame, end_frame, confidence])
    df = pd.DataFrame(trdf_rows, columns=['video_id', 'id', 'action', 'start', 'end', 'confidence'])
    df.reset_index()
    df.to_csv("PKUMMDv2Train_map.csv")
    df2 = pd.DataFrame(ttdf_rows, columns=['video_id', 'id', 'action', 'start', 'end', 'confidence'])
    df2.reset_index()
    df2.to_csv("PKUMMDv2Test_map.csv")


def mergePK():
    # path = "/home/siddiqui/Action_Biometrics/data"
    v1train = pd.read_csv("PKUMMDv1Train_map.csv")
    v1test = pd.read_csv("PKUMMDv1Test_map.csv")

    v2train = pd.read_csv("PKUMMDv2Train_map.csv")
    v2test = pd.read_csv("PKUMMDv2Test_map.csv")
    v2train["id"] += 66
    v2test["id"] += 66

    train = pd.concat([v1train, v2train])
    test = pd.concat([v1test, v2test])
    train.to_csv("MPKUMMDTrain_map.csv")
    test.to_csv("MPKUMMDTest_map.csv")

def processPKUMMD():
    new_df_rows = []
    targets = pd.read_excel("C://Users/nyles/Downloads/targets.xlsx")
    sublabels = dict(zip(targets["Index"], targets["Subject ID"]))
    path = "C://Users/ny525072/Downloads"
    #path = "/home/c3-0/datasets/PKUMMD/LABELS/Train_Label_PKU_final/"
    for text in os.listdir(path):
        filedata = pd.read_csv(f"{os.path.join(path, text)}", header=None)
        for row in filedata.to_numpy():
            action, start, end, confidence = row[0], row[1], row[2], row[3]
            sub_id = sublabels[int(text[:4])]
            new_df_rows.append([text[:-4], action, start, end, confidence, sub_id])
    new_df = pd.DataFrame(new_df_rows, columns=["video", "action", "start", "end", "confidence", "id"])
    new_df.reset_index()
    new_df.to_csv("PKUMMD_map.csv")

def renamefiles():
    path = "home/c3-0/datasets/MergedPKUMMD/RGB_VIDEO"
    for file in os.listdir(path):
        if len(file) > 11:
            os.rename(os.path.join(path, file), os.path.join(path, f"{file[:-10]}"))

def fixsubjectid():
    df = pd.read_csv("MPKUMMDTrain_map2.csv")
    df2 = pd.read_csv("MPKUMMDTest_map2.csv")
    new_df = df.values
    new_df2 = df2.values
    for row in new_df:
        row[1] -= 1
    for row in new_df2:
        row[1] -= 1
    df = pd.DataFrame(new_df, columns=['video_id', 'id', 'action', 'start', 'end', 'confidence'])
    df2 = pd.DataFrame(new_df2, columns=['video_id', 'id', 'action', 'start', 'end', 'confidence'])
    print(np.unique(df["id"]), np.unique(df2["id"]))
    df.to_csv("MPKUMMDTrain_map3.csv")
    df2.to_csv("MPKUMMDTest_map3.csv")

def fixPKindex():
    videos = []
    subjects = []
    actions = []
    data = {}
    for row in open("NTUTrain_map.csv", 'r').readlines()[1:]:
        if len(row.split(',')) == 6:
            video_id, subject, action, placeholder1, placeholder2, placeholder3 = row.split(',')
            videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
            if subject not in subjects:
                subjects.append(subject)
            if action not in actions:
                actions.append(action)
            if f"{subject}_{action}" not in data:
                data[f"{subject}_{action}"] = []
            data[f"{subject}_{action}"].append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
    for key in data.keys():
        subject, action = key.split("_")[0], key.split("_")[1]
        if int(action) == 18 or int(action) == 21:
            continue
        new_action = random.choice([diff_action for diff_action in data.keys() if diff_action.split("_")[1] != action and diff_action.split("_")[0] == subject])
        new_subject = random.choice([diff_sub for diff_sub in data.keys() if diff_sub.split("_")[1] == action and diff_sub.split("_")[0] != subject])
        same_subject_vid = random.choice(data[new_action])
        same_action_vid = random.choice(data[new_subject])
        print(video_id[0:4], video_id[:4])
        ss_video_id, ss_action = same_subject_vid[0], same_subject_vid[2]
        sa_video_id, sa_subject = same_action_vid[0], same_action_vid[1]


def small_NTU():
    path = "/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_+rgb"
    train = []
    test = []
    for row in os.listdir(path):
        if int(row[8:12]) not in range(1, 28) or row[16:20] in ['A050', 'A051', 'A052', 'A053', 'A054', 'A055', 'A056', 'A057', 'A058',
                          'A059', 'A060', 'A106', 'A107', 'A108', 'A109', 'A110', 'A111', 'A112',
                          'A113', 'A114', 'A115', 'A116', 'A117', 'A118', 'A119', 'A120']:
            continue
        else:
            s_num, cam_id, sub_id, rep_num, act_id = row[0:4], row[4:8], row[8:12], row[12:16], row[16:20]
            if sub_id in range(1, 20):
                train.append([row, s_num[1:], cam_id[1:], sub_id[1:], rep_num[1:], act_id[1:]])
            else:
                test.append([row, s_num[1:], cam_id[1:], sub_id[1:], rep_num[1:], act_id[1:]])
    df = pd.DataFrame(train, columns=['video_id', 'setup', 'camera', 'subject', 'repetition', 'action'])
    df.reset_index()
    df.to_csv("/home/siddiqui/Action_Biometrics/data/SmallNTUTrain_map.csv")

    df = pd.DataFrame(test, columns=['video_id', 'setup', 'camera', 'subject', 'repetition', 'action'])
    df.reset_index()
    df.to_csv("/home/siddiqui/Action_Biometrics/data/SmallNTUTest_map.csv")

def save():
    image_folder = 'C://Users/nyles/Downloads/test2'
    video_name = 'video3.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 24, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    
def vid_to_jpg():
    cap = cv2.VideoCapture("C://Users/nyles/Downloads/S001C001P001R001A001_rgb.avi")
    ret, frame = cap.read()
    print(ret)
    count = 1000
    while ret:
        print(ret)
        cv2.imwrite(f"C://Users/nyles/Downloads/test/{count}.jpg", frame)
        count += 1
        ret, frame = cap.read()

if __name__ == '__main__':
    save()


    # flag = False
    # global_counter = 0
    # counter = 0
    # for count, line in enumerate(open(r"C:\Users\ny525072\IdeaProjects\cache_simulator\MachineProblem1\debug_runs\debug6.txt", 'r').readlines(), 1):
    #     if flag:
    #         global_counter = line.split(" ")[1]
    #         flag = False
    #     if "-----" in line:
    #         flag = True
    #     if "L2 victim:" in line:
    #         if "dirty" in line:
    #             counter += 1
    # print(counter)

    # vr = VideoReader(os.path.join("/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb", "S014C002P025R002A045_rgb.avi"), height=270, width=480)
    # start_frame = 0
    # end_frame = len(vr)
    # frame_ids = np.linspace(start_frame, end_frame - 1, 32).astype(int)
    # frames = vr.get_batch(frame_ids)
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)
    #
    # start = timeit.default_timer()
    # vr = VideoReader(os.path.join("/home/c3-0/datasets/MergedPKUMMD/RGB_VIDEO", "0013-R.avi"),
    #                  height=270, width=480)
    # start_frame = 1602
    # end_frame = 1787
    # frame_ids = np.linspace(start_frame, end_frame - 1, 32).astype(int)
    # frames = vr.get_batch(frame_ids)
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)
    #
    # start = timeit.default_timer()
    # start_timestamp, end_timestamp = 12.1, 18
    # total_frames = len(os.listdir(os.path.join('/home/c3-0/datasets/Charades/Charades_v1_rgb', 'YSKX3')))
    # start_frame = max(1, int(start_timestamp * 24))
    # end_frame = min(total_frames, int(end_timestamp * 24))
    # frame_ids = np.linspace(start_frame, end_frame, 24).astype(int)
    # frames = []
    # for frame_id in frame_ids:
    #     f = os.path.join('/home/c3-0/datasets/Charades/Charades_v1_rgb', 'YSKX3', 'YSKX3' + '-' + str(frame_id).zfill(6) + '.jpg')
    #     frame = cv2.imread(f)
    #     if frame is None:
    #         continue
    #     frames.append(frame)
    # start_frame = 1602
    # end_frame = 1787
    # frame_ids = np.linspace(start_frame, end_frame - 1, 32).astype(int)
    # frames = vr.get_batch(frame_ids)
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)
    # path = "C://Users/ny525072/Downloads/"
    # traindf = pd.read_csv(os.path.join(path, "NTUTrain_map.csv"))
    # traindf.reset_index()
    # traindf.to_csv(os.path.join(path, "NTUTrain_map.csv"))


