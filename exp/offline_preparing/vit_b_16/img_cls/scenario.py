from data import build_scenario

scenario = build_scenario(
    source_datasets_name=['ImageNet', 'SYNSIGNS'],
    target_datasets_order=['Caltech256', 'GTSRB', 'DomainNet (real)'] * 10,
    da_mode='close_set',
    data_dirs={
        'ImageNet': '/data/zql/datasets/imagenet2012',
        'SYNSIGNS': '/data/zql/datasets/synthsign/synthetic_data/train_ImageFolder/',
        'Caltech256': '/data/zql/datasets/Caltech-256/data/caltech256/256_ObjectCategories/',
        'GTSRB': '/data/zql/datasets/GTSRB/GTSRB',
        'DomainNet (real)': '/data/zql/datasets/domain_net/real'
    },
    transforms={
        'ImageNet': None,
        'SYNSIGNS': None,
        'Caltech256': None,
        'GTSRB': None,
        'DomainNet (real)': None
    }
)