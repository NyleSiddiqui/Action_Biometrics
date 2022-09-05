import json

def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'ntu_rgbd_120':
        cfg.videos_folder =  '/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb'
        cfg.ignore_actions = ['A050', 'A051', 'A052', 'A053', 'A054', 'A055', 'A056', 'A057', 'A058', 
                              'A059', 'A060', 'A106', 'A107', 'A108', 'A109', 'A110', 'A111', 'A112', 
                              'A113', 'A114', 'A115', 'A116', 'A117', 'A118', 'A119', 'A120']
        cfg.train_subjects = range(0, 70)
        cfg.test_subjects = range(70, 106)
        cfg.num_subjects = 106
    elif dataset == "pkummd":
        cfg.videos_folder =  '/home/c3-0/datasets/PKUMMDv2/RGB_VIDEO_v2' 
        cfg.train_subjects = range(0, 10)
        cfg.test_subjects = range(10, 14)
        cfg.num_subjects = 13
    else:
        raise NotImplementedError
    cfg.dataset = dataset
    cfg.saved_models_dir = './results/saved_models'
    cfg.outputs_folder = './results/outputs'
    cfg.tf_logs_dir = './results/logs'
    return cfg