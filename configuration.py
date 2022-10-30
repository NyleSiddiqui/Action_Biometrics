import json

def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'ntu_rgbd_120':
        cfg.videos_folder =  '/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb'
        cfg.train_annotations = '/home/siddiqui/Action_Biometrics/data/NTUTrain_map.csv'
        cfg.test_annotations = '/home/siddiqui/Action_Biometrics/data/NTUTest_map.csv'
        cfg.ignore_actions = ['A050', 'A051', 'A052', 'A053', 'A054', 'A055', 'A056', 'A057', 'A058', 
                              'A059', 'A060', 'A106', 'A107', 'A108', 'A109', 'A110', 'A111', 'A112', 
                              'A113', 'A114', 'A115', 'A116', 'A117', 'A118', 'A119', 'A120']
        cfg.train_subjects = range(1, 71)
        cfg.test_subjects = range(71, 107)
        cfg.num_actions = 94
        cfg.num_subjects = 106
        
    elif dataset == "pkummd":
        cfg.videos_folder =  '/home/c3-0/datasets/MergedPKUMMD/RGB_VIDEO'
        cfg.train_annotations = '/home/siddiqui/Action_Biometrics/data/MPKUMMDTrain_map4edit.csv'
        cfg.test_annotations = '/home/siddiqui/Action_Biometrics/data/MPKUMMDTest_map3.csv'
        cfg.train_subjects = range(59)
        cfg.test_subjects = range(59, 75)
        cfg.num_subjects = 76
        cfg.num_actions = 43

    elif dataset == 'charades':
        cfg.videos_folder = '/home/c3-0/datasets/Charades/Charades_v1_rgb'
        cfg.train_annotations = "/home/siddiqui/Action_Biometrics/data/CharadesTrain_map.csv" 
        cfg.test_annotations =  "/home/siddiqui/Action_Biometrics/data/CharadesTest_map.csv"
        #cfg.fps = 24
        cfg.train_subjects = list(set([line.split(',')[1] for line in open(cfg.train_annotations, 'r').readlines()[1:]]))
        cfg.test_subjects = list(set([line.split(',')[1] for line in open(cfg.test_annotations, 'r').readlines()[1:]]))
        cfg.num_actions = 157
        
    else:
        raise NotImplementedError
    cfg.dataset = dataset
    cfg.saved_models_dir = './results/saved_models'
    cfg.outputs_folder = './results/outputs'
    cfg.tf_logs_dir = './results/logs'
    return cfg