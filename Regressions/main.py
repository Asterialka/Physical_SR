import os
import utils
import numpy as np
import pickle
from data import *
from evaluation_metrics import *
from rbf import RBFNet
import time
import matplotlib.pyplot as plt

regress_mode = {'type': 'rbf', # poly, root-poly or rbf
                'order': 1,
                'num_centers': 45,
                'dim_spec': 34,
                'dim_rgb': 3,
                'target_wavelength': np.arange(400,701,9)}

advanced_mode = {'Physically_Plausible': False,
                 'Data_Augmentation': False, # "[a, b], a = range, b = iteration", or, "False"
                 'Exposure': [0.5,1,2],
                 'Sparse': True}

operation_mode = {'Train': False,
                  'Validation': False,
                  'Test': True}

cost_funcs = {'val': mrae,
              'test': [mrae, dE00, dE00_illuminant_A, dE00_illuminant_D65,
                       dE00_camera_sony, dE00_camera_nikon, dE00_camera_canon, nse
                       ]}

test_modes = {'Mean': np.mean,
              'Pt99.9': lambda X: np.percentile(X, 99.9)
              }

directories = {'data': '/home/ashilov/Desktop/ippi/Physically_Plausible_Spectral_Reconstruction/',
               'precal': 'E:/UEA/Resource/ICVL/precal/',
               'sparse_label': 'E:/UEA/Resource/ICVL/sparse/',
               'models': './models/',
               'HSCNN-R': 'E:/UEA/Code/HSCNN-R/',
               'results': './results/',
               'resurce': './resources'}

resources = {'cmf': load_cie64cmf('./resources/',  regress_mode['target_wavelength']),
             'csf_s': load_camera_csf('./resources', 'sony_imx135', regress_mode['target_wavelength']),
             'csf_c': load_camera_csf('./resources', 'canon_5d', regress_mode['target_wavelength']),
             'csf_n': load_camera_csf('./resources', 'nikon_d810', regress_mode['target_wavelength']),
             'illuminant_A': load_illuminant('./resources/', 'A', regress_mode['target_wavelength']),
             'illuminant_E': load_illuminant('./resources/', 'E', regress_mode['target_wavelength']),
             'illuminant_D65': load_illuminant('./resources/', 'D65', regress_mode['target_wavelength']),
             'null_basis': np.load('./resources/2019-09-12_009_null_basis.npy'),
             'funda_mat': np.load('./resources/2019-09-12_009_fundamental_metamer.npy'),
             'rbf_net': [pickle.load(open('./resources/rbf_icvl_train'+i+'.pkl', 'rb')) for i in ['AB', 'CD', 'AB_aug', 'CD_aug']],
             'wp_list': loadmat2array('./white_point_data.h5', 'white_spectrum'),
             'name_list': open('./resources/fname_wp.txt').readlines(),
             'crsval_name_list': [open('./resources/fn_icvl_group_'+i+'.txt').readlines() for i in ['A','B','C','D']],
             'anchors': [loadmat2array('./resources/icvl_anchors.mat', 'anchor_'+ i) for i in ['A', 'B']]}


def train(img_list, crsval_mode=0):
    print("  Training Regression Matrix...")
    train_suffix, _ = generate_crsval_suffix(crsval_mode)
    if advanced_mode['Sparse']:
        # Loading pretrained sparse data
        assert os.path.isdir(directories['sparse_label']), 'Please run sparse.py first'
        with open(os.path.join(directories['precal'], 'sparse_all_data'+train_suffix+'.pkl'), 'rb') as handle:
            gt_data = pickle.load(handle)
        nearest_neighbors = np.load(os.path.join(directories['precal'], 'sparse_neighbor_idx'+train_suffix+'.npy')).astype(int)
        num_anchors, num_neighbors = nearest_neighbors.shape
        
        # Training "Multiple" Regression Matrix
        RegMat = []
        for i in range(num_anchors):
            RegMat.append(utils.RegressionMatrix(regress_mode, advanced_mode))
            
        for i in range(num_anchors):
            nearest_idx = nearest_neighbors[i, :]
            gt_data_nearest = {}
            gt_data_nearest['spec'] = gt_data['spec'][nearest_idx, :]
            gt_data_nearest['rgb'] = gt_data['rgb'][nearest_idx, :]
            
            regress_input, regress_output = utils.data_transform(gt_data_nearest, regress_mode, advanced_mode, crsval_mode, resources)
            RegMat[i].update(regress_input, regress_output)
    else:
        RegMat = utils.RegressionMatrix(regress_mode, advanced_mode)
        if advanced_mode['Data_Augmentation']:
            with open(os.path.join(directories['precal'], 'sparse_all_data'+train_suffix+'.pkl'), 'rb') as handle:
                gt_data = pickle.load(handle)
            
            for i in range(advanced_mode['Data_Augmentation'][1]):
                gt_data_aug = {}
                gt_data_aug['spec'] = utils.apply_random_scale(gt_data['spec'], base=advanced_mode['Data_Augmentation'][0])
                gt_data_aug['rgb'] =  gt_data_aug['spec'] @ resources['cmf']
                regress_input, regress_output = utils.data_transform(gt_data_aug, regress_mode, advanced_mode, crsval_mode, resources)
                RegMat.update(regress_input, regress_output)
        else:    
            for img_name in img_list:
                spec_img = load_icvl_data(directories['data'], img_name[:-1])
                gt_data = {}
                gt_data['spec'], gt_data['rgb'] = utils.cal_gt_data(spec_img, resources['cmf'], augment=advanced_mode['Data_Augmentation'])        
                regress_input, regress_output = utils.data_transform(gt_data, regress_mode, advanced_mode, crsval_mode, resources)
                RegMat.set_gamma(1e-5)  
                RegMat.update(regress_input, regress_output)
    print("regress mode = ", regress_mode['type'])
    utils.save_model(RegMat, directories['models'], regress_mode, advanced_mode, train_suffix)
    

def validate(img_list, crsval_mode=0):    
    train_suffix, val_suffix = generate_crsval_suffix(crsval_mode)    
    RegMat = utils.load_model(directories['models'], regress_mode, advanced_mode, train_suffix)
    
    if advanced_mode['Sparse']:
        # Loading precollected sparse data
        with open(os.path.join(directories['precal'], 'sparse_all_data'+val_suffix+'.pkl'), 'rb') as handle:
            gt_data = pickle.load(handle)
        nearest_neighbors = np.load(os.path.join(directories['precal'], 'sparse_neighbor_idx'+val_suffix+'.npy')).astype(int)
        num_anchors, num_neighbors = nearest_neighbors.shape
        
        print("  Validating Regression Matrix...")
        for i in range(num_anchors):
            if i%200 == 100:
                print('    anchor', i)
            nearest_idx = nearest_neighbors[i, :]
            gt_data_nearest = {}
            gt_data_nearest['spec'] = gt_data['spec'][nearest_idx, :]
            gt_data_nearest['rgb'] = gt_data['rgb'][nearest_idx, :]
            
            regress_input, _ = utils.data_transform(gt_data_nearest, regress_mode, advanced_mode, crsval_mode, resources)
            RegMat[i] = utils.regularize(RegMat[i], regress_input, gt_data_nearest, advanced_mode, 
                                         cost_funcs['val'], resources, show_graph=False)
        
        utils.save_model(RegMat, directories['models'], regress_mode, advanced_mode, train_suffix + val_suffix) 
        
    else:
        if advanced_mode['Data_Augmentation']:
            dir_precal_data = os.path.join(directories['precal'], 'all_data'+val_suffix+'_aug'+str(advanced_mode['Data_Augmentation'][0])+'.pkl')
        else:
            dir_precal_data = os.path.join(directories['precal'], 'all_data'+val_suffix+'.pkl')
        
        if os.path.isfile(dir_precal_data):
            with open(dir_precal_data, 'rb') as handle:
                gt_data = pickle.load(handle)
        else:
            print("  Preparing Validation Images...")
            gt_data = utils.collect_gt_data(directories['data'], img_list, resources['cmf'], 2000, augment=advanced_mode['Data_Augmentation'])    
            with open(dir_precal_data, 'wb') as handle:
                pickle.dump(gt_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print("  Validating Regression Matrix...")
        regress_input, _ = utils.data_transform(gt_data, regress_mode, advanced_mode, crsval_mode, resources)
        
        RegMat = utils.regularize(RegMat, regress_input, gt_data, advanced_mode, 
                                  cost_funcs['val'], resources, show_graph=False)
        
        utils.save_model(RegMat, directories['models'], regress_mode, advanced_mode, train_suffix + val_suffix)

def test_git(img_list, crsval_mode=0, file_name=()):
    print("  Testing...")
    train_suffix, val_suffix = generate_crsval_suffix(crsval_mode)  
    
    # Инициализация структур для хранения суммарных метрик
    num_exposures = len(advanced_mode['Exposure'])
    num_metrics = 8  # mrae, dE00, dE00_A, dE00_D65, dE00_sony, dE00_nikon, dE00_canon, nse
    total_metrics = np.zeros((num_exposures, num_metrics, 2))  # [exposure][metric][mean, pt99.9]
    img_count = 0
    
    if regress_mode['type'] == 'HSCNN-R':
        hscnn_model, hscnn_type = load_hscnn_R_model(directories['HSCNN-R'], regress_mode, advanced_mode, crsval_mode)
    else:
        RegMat = utils.load_model(directories['models'], regress_mode, advanced_mode, train_suffix)
    
    if advanced_mode['Sparse']:
        for img_name in img_list:
            img_count += 1
            print("    ", img_name[:-1])
            spec_img = load_icvl_data(directories['data'], img_name[:-1])
            gt_data = {} 
            gt_data['spec'], gt_data['rgb'] = utils.cal_gt_data(spec_img, resources['cmf'], augment=False)
            
            wp_spec = load_icvl_white_point(resources['wp_list'], resources['name_list'], img_name)
            nearest_anchor = np.load(os.path.join(directories['sparse_label'], img_name[:-5]+'_label.npy')).astype(int)
            active_anchors = np.unique(nearest_anchor).astype(int)
                
            gt_data = process_colors(gt_data, resources, wp_spec, cost_funcs['test'], regress_mode['target_wavelength'], exposure=1)
            
            write_row = [img_name]
            for exp_idx, exposure in enumerate(advanced_mode['Exposure']):                     
                regress_input, _ = utils.data_transform(gt_data, regress_mode, advanced_mode, crsval_mode, resources, exposure)
                num_data = regress_input.shape[0]
                recovery = {'spec': np.zeros((num_data, regress_mode['dim_spec'])),
                            'rgb': np.zeros((num_data, regress_mode['dim_rgb']))}  
                                
                for i in active_anchors:
                    is_nearest = nearest_anchor == i
                    rgb_nearest = gt_data['rgb'][is_nearest, :]
                    regress_input_nearest = regress_input[is_nearest, :]
                    recovery_part = utils.recover(RegMat[i].get_matrix(), regress_input_nearest, advanced_mode, resources, rgb_nearest, exposure)

                    recovery['spec'][is_nearest, :] = recovery_part['spec']
                    recovery['rgb'][is_nearest, :] = recovery_part['rgb']
                
                recovery = process_colors(recovery, resources, wp_spec, cost_funcs['test'], regress_mode['target_wavelength'], exposure, gt_data)
                
                for cost_idx, cost_func in enumerate(cost_funcs['test']):
                    cost = cost_func(gt_data, recovery, exposure) 
                    for tmode_idx, (tmode_key, tmode_func) in enumerate(test_modes.items()):
                        metric_value = tmode_func(cost)
                        write_row.append(metric_value)
                        
                        # Для каждой метрики сохраняем только mean и pt99.9 (tmode_idx 0 и 1)
                        if tmode_idx < 2:
                            total_metrics[exp_idx, cost_idx, tmode_idx] += metric_value

            if len(file_name):
                print("\nИтоговые метрики для изображения:", img_name)
                for i, exposure in enumerate(advanced_mode['Exposure']):
                    print(f"\nЭкспозиция: {exposure}x")
                    for j, cost_name in enumerate(['mrae', 'dE00', 'dE00_A', 'dE00_D65', 'dE00_sony', 'dE00_nikon', 'dE00_canon', 'nse']):
                        mean_val = write_row[1 + i*16 + j*2] 
                        pt_val = write_row[1 + i*16 + j*2 + 1]
                        print(f"{cost_name}: Mean = {mean_val:.4f}, Pt99.9 = {pt_val:.4f}")
                write2csvfile(file_name, write_row)
    else:
        for img_name in img_list:
            img_count += 1
            spec_img = load_icvl_data(directories['data'], img_name[:-1])
            gt_data = {} 
            gt_data['spec'], gt_data['rgb'] = utils.cal_gt_data(spec_img, resources['cmf'], augment=False)

            wp_spec = resources['illuminant_D65']
            gt_data = process_colors(gt_data, resources, wp_spec, cost_funcs['test'], regress_mode['target_wavelength'], exposure=1) 
            
            write_row = [img_name]
            for exp_idx, exposure in enumerate(advanced_mode['Exposure']):
                if regress_mode['type'] == 'HSCNN-R':
                    recovery = {}
                    recovery = utils.recover_HSCNN_R(gt_data['rgb'], spec_img.shape, resources, hscnn_model, hscnn_type, exposure)
                else:
                    regress_input, _ = utils.data_transform(gt_data, regress_mode, advanced_mode, crsval_mode, resources, exposure)
                    recovery = {}
                    recovery = utils.recover(RegMat.get_matrix(), regress_input, advanced_mode, resources, gt_data['rgb'], exposure) 
  
                recovery = process_colors(recovery, resources, wp_spec, cost_funcs['test'], regress_mode['target_wavelength'], exposure, gt_data)                
                gt_data['spec'] = gt_data['spec'].clip(min=1e-8)

                for cost_idx, cost_func in enumerate(cost_funcs['test']):
                    cost = cost_func(gt_data, recovery, exposure)
                    for tmode_idx, (tmode_key, tmode_func) in enumerate(test_modes.items()):
                        metric_value = tmode_func(cost)
                        write_row.append(metric_value)
                        
                        # Для каждой метрики сохраняем только mean и pt99.9 (tmode_idx 0 и 1)
                        if tmode_idx < 2:
                            total_metrics[exp_idx, cost_idx, tmode_idx] += metric_value
            
            if len(file_name):
                print("\nИтоговые метрики для изображения:", img_name)
                for i, exposure in enumerate(advanced_mode['Exposure']):
                    print(f"\nЭкспозиция: {exposure}x")
                    for j, cost_name in enumerate(['mrae', 'dE00', 'dE00_A', 'dE00_D65', 'dE00_sony', 'dE00_nikon', 'dE00_canon', 'nse']):
                        mean_val = write_row[1 + i*16 + j*2] 
                        pt_val = write_row[1 + i*16 + j*2 + 1]
                        print(f"{cost_name}: Mean = {mean_val:.4f}, Pt99.9 = {pt_val:.4f}")
                write2csvfile(file_name, write_row)
    
    # Вывод средних метрик по всем изображениям
    if img_count > 0:
        print("\nСредние метрики по всему тесту:")
        total_metrics /= img_count
        for exp_idx, exposure in enumerate(advanced_mode['Exposure']):
            print(f"\nЭкспозиция: {exposure}x")
            for cost_idx, cost_name in enumerate(['mrae', 'dE00', 'dE00_A', 'dE00_D65', 'dE00_sony', 'dE00_nikon', 'dE00_canon', 'nse']):
                mean_val = total_metrics[exp_idx, cost_idx, 0]
                pt_val = total_metrics[exp_idx, cost_idx, 1]
                print(f"{cost_name}: Mean = {mean_val:.4f}, Pt99.9 = {pt_val:.4f}")
        
        # Запись средних метрик в файл
        if len(file_name):
            avg_row = ["Average"]
            for exp_idx in range(num_exposures):
                for cost_idx in range(num_metrics):
                    avg_row.append(total_metrics[exp_idx, cost_idx, 0])  # mean
                    avg_row.append(total_metrics[exp_idx, cost_idx, 1])  # pt99.9
            write2csvfile(file_name, avg_row)

if __name__ == '__main__':

    folder_path = "data"
    folder_path_1 = "test_images"
    img_list = ['test_images/' + f + '\n' for f in os.listdir(folder_path_1) if os.path.isfile(os.path.join(folder_path_1, f))]
    train_img_list = ['data/' + f + '\n' for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    operation_mode['Train'] = True
    operation_mode['Validation'] = False
    operation_mode['Test'] = True
    advanced_mode['Data_Augmentation'] = False 
    advanced_mode['Sparse'] = False  
    regress_mode['type'] = 'rbf'

    train(train_img_list, crsval_mode=1)
    if operation_mode['Test']:
        file_name = initialize_csvfile(directories['results'], regress_mode, advanced_mode, cost_funcs, test_modes)
    
    test_git(img_list, crsval_mode=1, file_name=file_name) 