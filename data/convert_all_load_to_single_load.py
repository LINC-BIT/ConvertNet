"""
convert load-all-images-into-memory-before-training dataset
to load-when-training-dataset


"""


from torchvision.datasets import CIFAR10, STL10, MNIST, USPS, SVHN
import os
import tqdm


def convert(datasets_of_split, new_dir):
    img_idx = {}

    for d in datasets_of_split:
        for x, y in tqdm.tqdm(d, total=len(d), dynamic_ncols=True):
            # print(type(x), type(y))
            # break
            # y = str(y)
            if y not in img_idx:
                img_idx[y] = -1
            img_idx[y] += 1

            p = os.path.join(new_dir, f'{y:06d}', f'{img_idx[y]:06d}' + '.png')
            os.makedirs(os.path.dirname(p), exist_ok=True)

            x.save(p)


if __name__ == '__main__':
    # convert(
    #     [CIFAR10('/data/zql/datasets/CIFAR10', True, download=True), CIFAR10('/data/zql/datasets/CIFAR10', False, download=True)],
    #     '/data/zql/datasets/CIFAR10-single'
    # )

    # convert(
    #     [STL10('/data/zql/datasets/STL10', 'train', download=False), STL10('/data/zql/datasets/STL10', 'test', download=False)],
    #     '/data/zql/datasets/STL10-single'
    # )

    # convert(
    #     [MNIST('/data/zql/datasets/MNIST', True, download=True), MNIST('/data/zql/datasets/MNIST', False, download=True)],
    #     '/data/zql/datasets/MNIST-single'
    # )

    convert(
        [SVHN('/data/zql/datasets/SVHN', 'train', download=True), SVHN('/data/zql/datasets/SVHN', 'test', download=True)],
        '/data/zql/datasets/SVHN-single'
    )

    # convert(
    #     [USPS('/data/zql/datasets/USPS', True, download=False), USPS('/data/zql/datasets/USPS', False, download=False)],
    #     '/data/zql/datasets/USPS-single'
    # )