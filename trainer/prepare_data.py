#import libraries
from collections import Counter
import tensorflow as tf
import numpy as np
import pickle


def prep_data(dataset_path, test_size):
    #get list of categories from dataset directory
    print(dataset_path)
    print(tf.gfile.ListDirectory(dataset_path))
    categories = [x.replace('/','') for x in tf.gfile.ListDirectory(dataset_path)]
    if '.DS_Store' in categories:
        categories.remove('.DS_Store')
    #initialize encoder and img file masterlist
    enc, img_master, label_master = {}, [], []
    for i, cat in enumerate(categories):
        enc[cat] = i #set class encoder key:value
        cat_img = ['{}/{}/{}'.format(dataset_path,cat,x) for x in tf.gfile.ListDirectory(dataset_path+'/'+cat)]
        img_master = img_master + cat_img
        label_master = label_master + [i]*len(cat_img)
    #calculate class frequency
    cat_freq = Counter(label_master)

    #balance class frequency
    sample_target = np.max(list(cat_freq.values()))
    for i, cat in enumerate(categories):
        if cat_freq[i]<sample_target:
            n_sample = sample_target-cat_freq[i]
            sample_list = ['{}/{}/{}'.format(dataset_path,cat,x) for x in tf.gfile.ListDirectory(dataset_path+'/'+cat)]
            img_sample = np.random.choice(sample_list, size=n_sample, replace=True)
            #oversample via random sampling
            img_master = img_master + img_sample.tolist()
            label_master = label_master + [i]*n_sample

    #perform train/test split
    train_indices, test_indices = [], []
    for i in cat_freq.keys():
        #get list of all indices where label falls in the category
        temp_indices = np.where(np.array(label_master)==i)[0]
        #get test_indices
        temp_test = np.random.choice(temp_indices,size=int(len(temp_indices)*test_size),replace=False)
        temp_train = temp_indices[~np.isin(temp_indices, temp_test)]
        #add to train/test indices    
        train_indices += temp_train.tolist()
        test_indices += temp_test.tolist()
    img_train = np.array(img_master)[train_indices].tolist()
    label_train = np.array(label_master)[train_indices].tolist()
    img_test  = np.array(img_master)[test_indices].tolist()
    label_test = np.array(label_master)[test_indices].tolist()
    return {'train':{'img':img_train, 'label':label_train}, 
            'test':{'img':img_test, 'label':label_test}}

      