import os
import numpy as np


def gen_data(path, k, dir):
    files = os.listdir(path)
    s = []
    weight_list = None
    label_list = None
    instance_list = []
    file_list = []
    print(len(files))
    ss = 0
    for file in files:
        if not os.path.isdir(file):
            print(file)
            # 这里要加编码方式
            f = open(path + "/" + file, 'r', encoding='utf-8')
            iter_f = iter(f)
            str = ""
            idx = 0
            target = -1
            weight = []
            label = []
            num = 0
            for line in iter_f:
                if idx > 0 and idx % k == 1:
                    inst = line.split(',')
                    inst = inst[0:-1]
                    instance = []
                    for feature in inst:
                        instance.append(float(feature))

                    target = instance[0]
                    if target >= 0.0:
                        weight.append(target + 1.0)
                        label.append(1.0)
                        instance_list.append(instance)
                        file_list.append(file)
                        num += 1
                idx += 1
            if num == 1:
                inst = line.split(',')
                inst = inst[0:-1]
                instance = []
                for feature in inst:
                    instance.append(float(feature))

                target = instance[0]
                if target >= 0.0:
                    weight.append(target + 1.0)
                    label.append(1.0)
                    instance_list.append(instance)
                    file_list.append(file)

            weight = np.array(weight)
            label = np.array(label)
            if target == 0.0:
                print(file)

            weight /= (target + 1.0)
            label *= target

            if weight_list is None:
                weight_list = weight
            else:
                weight_list = np.concatenate((weight_list, weight), axis=0)
            if label_list is None:
                label_list = label
            else:
                label_list = np.concatenate((label_list, label), axis=0)

    instance_list = np.array(instance_list)

    np.save(dir + "/instance.npy", instance_list)
    np.save(dir + "/target.npy", label_list)
    np.save(dir + "/weight.npy", weight_list)
    np.save(dir + "/file.npy", file_list)


if __name__ == '__main__':
    # gen_data("data_samples/train", 200, "train")
    # gen_data("data_samples/test1", 200, "test")
    weight = np.load("train/weight.npy")
    print(weight)
    weight = np.load("test/weight.npy")
    print(weight)
