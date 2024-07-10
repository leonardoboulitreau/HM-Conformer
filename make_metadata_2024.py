import os

def read_metadata_ASVspoof2024(path_meta):
    """
    Read metadata -> dictionary
    key: file_name, items: infomation
    """
    metadata = {}
    cnt = 0
    for line in open(path_meta).readlines():
        strI = line.replace('\n', '').split(' ')
        metadata[strI[1]] = line
        cnt += 1
    print(f'lines: {cnt}')
    return metadata

def write_metadata(
        org_metadata,
        path,
        option,
        exception=[],
        codecs=["flac"]
    ):
    path_train = path + '/train'
    path_write = path + '/metadata_trn_spd.txt'
    with open(path_write, 'w') as f:
        for codec in codecs:
            _path_train = path_train + "/" + codec
            for root, _, files in os.walk(_path_train):
                for file in files:
                    if '.flac' in file:
                        f_dir = root.split('/')[-1]     # flac
                        f_name = file.split('.')[0]     # LA_T_*
                        if f_name in exception:
                            print('exception file')
                            continue
                        org_name = f_name[:12]  # LA_T_0000000
                        
                        line = org_metadata[org_name]
                        new_line = line.replace(org_name, f_dir + '/' + f_name)
                        f.write(new_line)

        
if __name__ == '__main__':
    YOUR_ASVspoof2024_PATH = '/data/ASVspoof2024'
    path_meta_trn = YOUR_ASVspoof2024_PATH + '/ASVspoof5.train.metadata.txt'
    metadata = read_metadata_ASVspoof2024(path_meta_trn)
    write_metadata(metadata, YOUR_ASVspoof2024_PATH, 'trn')