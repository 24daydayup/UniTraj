args = {
    'data': {
        'dataset': 'worldtrace',
        'traj_length': 200,
        'emb_dim': 128,
        'num_workers': 24,
    },
    'training': {
        'batch_size': 1024,
        'n_epochs': 1000,
    },
    'rag': {
        'enable': True,
        'k': 256,                     # 知识库存的均值轨迹个数
        'kb_path': 'data/kb_means.pt',# 知识库保存/加载路径
        'topk': 3,                    # 检索 Top-K
        'temperature': 0.07,          # 注意力温度
        'inject_prior_in_train': False,
        'inject_prior_in_sample': True
    },
}
