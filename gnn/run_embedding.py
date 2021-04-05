import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model
from embedding_model import sample_neighs, OurMethod
from utils import preprocess_adj, plot_embeddings, load_data_v1
import time

if __name__ == "__main__":
    # 替换cora以加载新的数据集
    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data_v1(
        'cora', "../data/cora/")

    features /= features.sum(axis=1, ).reshape(-1, 1)

    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())

    A = preprocess_adj(A)

    indexs = np.arange(A.shape[0])
    # 每层采样邻居的个数，下面表示第一层采样邻居为10，第二层为25，修改[10, 25]从而取样更多层，如：[10, 25，25，25].同时取消embdedding_model 269，270的注释.
    neigh_number = [10, 25]
    neigh_maxlen = []

    model_input = [features, np.asarray(indexs, dtype=np.int)]

    for num in neigh_number:
        sample_neigh, sample_neigh_len = sample_neighs(
            G, indexs, num, self_loop=False)
        model_input.extend([sample_neigh])
        neigh_maxlen.append(max(sample_neigh_len))

    model = OurMethod(feature_dim=features.shape[1],
                      neighbor_num=neigh_maxlen,
                      n_hidden=16,
                      n_classes=y_train.shape[1],
                      use_bias=True,
                      activation=tf.nn.relu,
                      aggregator_type='mean',
                      dropout_rate=0.5, l2_reg=2.5e-4)
    model.compile(Adam(0.01), 'categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])
    val_data = (model_input, y_val, val_mask)
    mc_callback = ModelCheckpoint('./best_model.h5',
                                  monitor='val_weighted_categorical_crossentropy',
                                  save_best_only=True,
                                  save_weights_only=True)

    print("start training")
    start_time = time.time()
    model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data,
              batch_size=A.shape[0], epochs=200, shuffle=False, verbose=2,
              callbacks=[mc_callback])
    end_time = time.time()
    print("embedding time cost: ", (end_time - start_time)/20)
    # exit()

    print("start loading weights")
    model.load_weights('./best_model.h5')
    eval_results = model.evaluate(
        model_input, y_test, sample_weight=test_mask, batch_size=A.shape[0])
    print('Done.\n'
          'Test loss: {}\n'
          'Test weighted_loss: {}\n'
          'Test accuracy: {}'.format(*eval_results))
