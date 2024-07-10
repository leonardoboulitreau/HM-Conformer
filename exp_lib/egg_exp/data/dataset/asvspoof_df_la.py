import os

from ._dataclass import DF_Item

class ASVspoof2021_DF_LA:
    NUM_TEST_ITEM    = 533928
    NUM_TEST_ITEM_LA = 148176

    PATH_TRAIN_TRL  = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    PATH_TRAIN_TRL_DA = 'LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA.txt'
    PATH_TRAIN_TRL_DA_wo_speed  = 'LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA_wo_speed.txt'
    
    PATH_DEV_TRL    = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'  
    PATH_DEV_TRL_DA = 'LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA_DEV.txt'
    PATH_DEV_TRL_DA_wo_speed    = 'LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA_DEV_wo_speed.txt'
    
    PATH_TRAIN_FLAC = 'LA/ASVspoof2019_LA_train'
    PATH_DEV_FLAC   = 'LA/ASVspoof2019_LA_dev'
    
    PATH_TEST_TRL   = 'keys/DF/CM/trial_metadata.txt'
    PATH_TEST_FLAC  = 'ASVspoof2021_DF_eval/flac'
    
    PATH_TEST_TRL_LA = 'keys/LA/CM/trial_metadata.txt'
    PATH_TEST_FLAC_LA = 'flac'
    
    PATH_TRAIN_2024 = 'metadata_trn.txt'
    PATH_TRAIN_2024_SPEED = 'metadata_trn_spd.txt'
    PATH_TRAIN_2024_FLAC = 'train'

    PATH_DEV_2024 = 'ASVspoof5.dev.metadata.txt'
    PATH_DEV_2024_FLAC = 'flac_D'
    
    PATH_EVAL_2024 = 'ASVspoof5.track_1.progress.trial.txt'
    PATH_EVAL_2024_FLAC = 'flac_E_prog'

    
    def __init__(self, path_train, path_test, path_test_LA=None, use_dev=True, DA_codec=False, DA_speed=True, print_info=False, path_dev_2024 = None, path_eval_2024 = None, train_2024=True):   
        self.train_set = []
        self.test_set = []
        self.test_set_LA = []
        self.train_set_2024 = []
        self.dev_set_2024 = []
        self.eval_set_2024 = []
        self.traintest_set_2024 = []
        self.class_weight = []

        if train_2024:
            if DA_codec:
                assert False
            # train_set 2024
            train_num_pos = 0
            train_num_neg = 0
            if DA_speed:
                print('> Using Speed Augmentation')
                trl = os.path.join(path_train, self.PATH_TRAIN_2024_SPEED)
            else:
                print('> NOT Using Speed Augmentation')
                trl = os.path.join(path_train, self.PATH_TRAIN_2024)
            for line in open(trl).readlines():
                strI = line.replace('\n', '').split(' ')
                f = os.path.join(path_train, self.PATH_TRAIN_2024_FLAC, f'{strI[1]}.flac') # Mexer aqui se for implementar codec aug
                attack_type = strI[4]
                spk_id = strI[0]
                label = 1 if strI[5] == 'bonafide' else 0   # Real: 1, Fake: 0
                if label == 0:
                    train_num_neg += 1
                else:
                    train_num_pos += 1
                item = DF_Item(f, label, attack_type, is_fake=(label == 0), spk_id = spk_id)
                self.train_set_2024.append(item)
            print('Training Set with', len(self.train_set_2024), ' files.')

        else:
            assert train_2024, 'o traintest tá assumindo que o path é de 2024'
            # train_set
            train_num_pos = 0
            train_num_neg = 0
            trl = os.path.join(path_train, self.PATH_TRAIN_TRL_DA if DA else self.PATH_TRAIN_TRL)  
            if not DA_speed: 
                trl = os.path.join(path_train, self.PATH_TRAIN_TRL_DA_wo_speed)
            for line in open(trl).readlines():
                strI = line.replace('\n', '').split(' ')
                if DA:
                    f = os.path.join(path_train, self.PATH_TRAIN_FLAC, f'{strI[1]}.flac')
                else: 
                    f = os.path.join(path_train, self.PATH_TRAIN_FLAC, f'flac/{strI[1]}.flac')
                attack_type = strI[3]
                label = 0 if strI[4] == 'bonafide' else 1   # Real: 0, Fake: 1
                if label == 0:
                    train_num_neg += 1
                else:
                    train_num_pos += 1
                item = DF_Item(f, label, attack_type, is_fake=(label == 1))
                self.train_set.append(item)
                
            # use dev_set in train
            if use_dev:
                trl_dev = os.path.join(path_train, self.PATH_DEV_TRL_DA if DA else self.PATH_DEV_TRL)
                if not DA_speed: 
                    trl_dev = os.path.join(path_train, self.PATH_DEV_TRL_DA_wo_speed)
                for line in open(trl_dev).readlines():
                    strI = line.replace('\n', '').split(' ')
                    if DA:
                        f = os.path.join(path_train, self.PATH_DEV_FLAC, f'{strI[1]}.flac')
                    else: 
                        f = os.path.join(path_train, self.PATH_DEV_FLAC, f'flac/{strI[1]}.flac')
                    attack_type = strI[3]
                    label = 0 if strI[4] == 'bonafide' else 1   # Real: 0, Fake: 1
                    if label == 0:
                        train_num_neg += 1
                    else:
                        train_num_pos += 1
                    item = DF_Item(f, label, attack_type, is_fake=(label == 1))
                    self.train_set.append(item)
        
        self.class_weight.append((train_num_neg + train_num_pos) / train_num_neg)
        self.class_weight.append((train_num_neg + train_num_pos) / train_num_pos)
        
        # test_set
        #test_num_pos = 0
        #test_num_neg = 0
        #trl = os.path.join(path_test, self.PATH_TEST_TRL)
        #for line in open(trl).readlines():
        #    strI = line.replace('\n', '').split(' ')
        #    # check subset
        #    if strI[7] != 'eval':
        #        continue
        #    f = os.path.join(path_test, self.PATH_TEST_FLAC, f'{strI[1]}.flac')
        #    attack_type = strI[4]
        #    label = 0 if attack_type == '-' else 1
        #    if label == 0:
        #        test_num_neg += 1
        #    else:
        #        test_num_pos += 1
        #        
        #    item = DF_Item(f, label, attack_type, is_fake=label == 1)
        #    self.test_set.append(item)
        # error check
        #assert len(self.test_set) == self.NUM_TEST_ITEM, f'[DATASET ERROR] - TEST_SAMPLE: {len(self.test_set)}, EXPECTED: {self.NUM_TEST_ITEM}'

        # test_set_LA
        #if path_test_LA is not None:
        #    test_num_pos_LA = 0
        #    test_num_neg_LA = 0
        #    trl_LA = os.path.join(path_test_LA, self.PATH_TEST_TRL_LA)
        #    for line in open(trl_LA).readlines():
        #        strI = line.replace('\n', '').split(' ')
        #        # check subset
        #        if strI[7] != 'eval':
        #            continue
        #        f = os.path.join(path_test_LA, self.PATH_TEST_FLAC_LA, f'{strI[1]}.flac')
        #        attack_type = strI[4]
        #        label = 0 if attack_type == 'bonafide' else 1
        #        if label == 0:
        #            test_num_neg_LA += 1
        #        else:
        #            test_num_pos_LA += 1
        #            
        #        item = DF_Item(f, label, attack_type, is_fake=label == 1)
        #        self.test_set_LA.append(item)
        #    # error check
        #    assert len(self.test_set_LA) == self.NUM_TEST_ITEM_LA, f'[DATASET ERROR] - TEST_SAMPLE: {len(self.test_set_LA)}, EXPECTED: {self.NUM_TEST_ITEM_LA}'
        
        # dev_set_2024
        dev2024_num_pos = 0
        dev2024_num_neg = 0
        trl = os.path.join(path_dev_2024, self.PATH_DEV_2024)
        for line in open(trl).readlines():
            strI = line.replace('\n', '').split(' ')
            f = os.path.join(path_dev_2024, self.PATH_DEV_2024_FLAC, f'{strI[1]}.flac')
            attack_type = strI[4]
            spk_id = strI[0]
            label = 1 if attack_type == 'bonafide' else 0
            if label == 0:
                dev2024_num_pos += 1
            else:
                dev2024_num_neg += 1
                
            item = DF_Item(f, label, attack_type, is_fake=label == 0, spk_id=spk_id)
            self.dev_set_2024.append(item)
        print('Development Set with', len(self.dev_set_2024), ' files.')

        # traintest_set_2024
        traintest_num_pos = 0
        traintest_num_neg = 0
        trl = os.path.join(path_train, self.PATH_TRAIN_2024)
        for line in open(trl).readlines():
            strI = line.replace('\n', '').split(' ')
            f = os.path.join(path_train, self.PATH_TRAIN_2024_FLAC, f'{strI[1]}.flac')
            attack_type = strI[4]
            spk_id = strI[0]
            label = 1 if attack_type == 'bonafide' else 0
            if label == 0:
                traintest_num_neg += 1
            else:
                traintest_num_pos += 1
            item = DF_Item(f, label, attack_type, is_fake=label == 0, spk_id=spk_id)
            self.traintest_set_2024.append(item)
        print('TrainTest Set with', len(self.traintest_set_2024), ' files.')

        # eval_set_2024
        eval_num_pos = 0
        eval_num_neg = 0
        trl = os.path.join(path_eval_2024, self.PATH_EVAL_2024)
        for line in open(trl).readlines():
            name = line.replace('\n', '')
            f = os.path.join(path_eval_2024, self.PATH_EVAL_2024_FLAC, f'{name}.flac')
            label = 0 
            attack_type = 'bonafide'
            spk_id = 'UNK'                
            item = DF_Item(f, label, attack_type, is_fake=label == 1, spk_id=spk_id)
            self.eval_set_2024.append(item)
        assert len(self.eval_set_2024) == 40765, print(len(self.eval_set_2024))
        print('Evaluation Set with', len(self.eval_set_2024), ' files.')

        # error check
        #assert len(self.test_set) == self.NUM_TEST_ITEM, f'[DATASET ERROR] - TEST_SAMPLE: {len(self.test_set)}, EXPECTED: {self.NUM_TEST_ITEM}'