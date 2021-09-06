import os

def generate_iamgets():
    trainset = r'/SWIM/ImageSets/Main/train.txt'
    valset = r'/SWIM/ImageSets/Main/val.txt'
    testset = r'/SWIM/ImageSets/Main/test.txt'
    img_dir = r'/SWIM/JPEGImages/'
    label_dir = r'/SWIM/Annotations/'
    root_dir = r'/SWIM/'

    for dataset in [trainset, valset, testset]:
        with open(dataset, 'r') as f:
            names = f.readlines()
            paths = [os.path.join(img_dir, x.strip() + '.jpg\n') for x in names]
            with open(os.path.join(root_dir, os.path.split(dataset)[1]), 'w') as fw:
                fw.write(''.join(paths))


if __name__ == '__main__':
    generate_iamgets()
