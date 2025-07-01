import numpy as np

from pathlib import Path

def main():
    target_dir = 'data/PathMNISTEmbeddings/smsl_embeddings'
    train_base_dir = '/'.join(target_dir.split('/')[:-1]) \
        + '/' + 'pca_embeddings/pathmnist_train_embeddings.npy'
    test_base_dir = '/'.join(target_dir.split('/')[:-1]) \
        + '/' + 'pca_embeddings/pathmnist_test_embeddings.npy'

    train_data = np.load(Path(train_base_dir))
    test_data = np.load(Path(test_base_dir))

    train_full_tag_train_data = np.append(train_data, np.ones((train_data.shape[0], 1), dtype=np.int32), 1)
    train_full_tag_test_data = np.append(test_data, np.zeros((test_data.shape[0], 1), dtype=np.int32), 1)

    test_full_tag_test_data = np.append(test_data, np.ones((test_data.shape[0], 1), dtype=np.int32), 1)
    test_full_tag_train_data = np.append(train_data, np.zeros((train_data.shape[0], 1), dtype=np.int32), 1)

    train_full_tag_full_data = np.append(train_full_tag_train_data, train_full_tag_test_data, 0)
    test_full_tag_full_data = np.append(test_full_tag_test_data, test_full_tag_train_data, 0)

    with open(Path(target_dir + "/pathmnist_smsl_train_pca_embeddings.npy"), "wb") as f:
        np.save(f, train_full_tag_train_data)
        f.close()

    with open(Path(target_dir + "/pathmnist_smsl_test_pca_embeddings.npy"), "wb") as f:
        np.save(f, test_full_tag_test_data)
        f.close()


    with open(Path(target_dir + "/pathmnist_smsl_train_full_pca_embeddings.npy"), "wb") as f:
        np.save(f, train_full_tag_full_data)
        f.close()

    with open(Path(target_dir + "/pathmnist_smsl_test_full_pca_embeddings.npy"), "wb") as f:
        np.save(f, test_full_tag_full_data)
        f.close()

if __name__ == "__main__":
    main()