import scipy
from scipy import signal
import os
import lmdb
import pickle

root_dir = 'C:\\Users\\lvxiangyu11\\workspace\\CBraMod\\dataset_dir_original\\SHU\\mat_files\\mat'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)
# print(files)

files_dict = {
    'train':files[:75],
    'val':files[75:100],
    'test':files[100:],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}
db = lmdb.open('C:\\Users\\lvxiangyu11\\workspace\\CBraMod\\dataset_processed\\SHU', map_size=3110612736)
for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        data = scipy.io.loadmat(os.path.join(root_dir, file))
        eeg = data['data']
        labels = data['labels'][0]
        bz, ch_num, points = eeg.shape
        print(eeg.shape)
        eeg_resample = signal.resample(eeg, 800, axis=2)
        eeg_ = eeg_resample.reshape(bz, ch_num, 4, 200)
        print(eeg_.shape, labels.shape)
        for i, (sample, label) in enumerate(zip(eeg_, labels)):
            sample_key = f'{file[:-4]}-{i}'
            # print(sample_key)
            data_dict = {
                'sample':sample, 'label':label-1
            }
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            txn.commit()
            dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()