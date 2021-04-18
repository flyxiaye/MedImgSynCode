import numpy as np

def rm_nan_ct(src_data):
    src_data = np.where(np.isnan(src_data), -1024, src_data)
    return src_data

def rm_neg(src_data):
    src_data = np.asarray(src_data)
    src_data = src_data - src_data.min()
    return src_data

def rm_max(src_data, ratio=0.01):
    """
    去除前0.01的像素点
    """
    data = np.copy(src_data)
    data = data.reshape(-1)
    data = np.sort(data)
    idx = int(data.shape[0] * (1 - ratio))
    data_max = data[idx]
    # print(data_max)
    src_data = np.where(src_data > data_max, data_max, src_data)
    # print(np.max(src_data))
    return np.asarray(src_data)


# 数据处理 按顺序切割并返回结果
def extract_ordered_overlap(full_imgs, patch_shape, stride_shape):
    patch_shape = np.asarray(patch_shape)
    stride_shape = np.asarray(stride_shape)
    # max_img_shape = np.max(full_imgs.shape)
    n_patch = (full_imgs.shape - patch_shape) // stride_shape + 2
    # print(n_patch)
    padded_shape = (n_patch - 1) * stride_shape + patch_shape
    # print(padded_shape)
    pad_img = np.zeros(shape=padded_shape)
    pad_img[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:full_imgs.shape[2]] = full_imgs
    for d in range(n_patch[0]):
        for h in range(n_patch[1]):
            for w in range(n_patch[2]):
                tmp_img = pad_img[d*stride_shape[0]: d*stride_shape[0]+patch_shape[0],
                                h*stride_shape[1]: h*stride_shape[1]+patch_shape[1],
                                w*stride_shape[2]: w*stride_shape[2]+patch_shape[2]]
                yield tmp_img


# 按顺序合成
def recompone_overlap(preds, img_shape, stride_shape):
    preds = np.asarray(preds)
    img_shape = np.asarray(img_shape)
    stride_shape = np.asarray(stride_shape)
    assert (len(preds.shape)==5)  # 检查张量尺寸
    assert (preds.shape[1]==1 or preds.shape[1]==3)
    patch_shape = preds.shape[2:]
    n_patch = (img_shape - patch_shape) // stride_shape + 2
    # N_patches_h = (img_h-patch_h)//stride_h+1 # img_h方向包括的patch_h数量
    # N_patches_w = (img_w-patch_w)//stride_w+1 # img_w方向包括的patch_w数量
    n_patch_imgs = np.prod(n_patch)
    # N_patches_img = N_patches_h * N_patches_w # 每张图像包含的patch的数目
    # assert (preds.shape[0]%N_patches_img==0   
    N_full_imgs = preds.shape[0]//n_patch_imgs # 全幅图像的数目
    full_img_shape = (n_patch - 1) * stride_shape + patch_shape
    full_img_shape = np.concatenate(((N_full_imgs,preds.shape[1]), full_img_shape))
    # print(full_img_shape)
    full_prob = np.zeros(full_img_shape)
    full_sum = np.zeros(full_img_shape)
    # print(patch_shape)
    # print(stride_shape)
    # print(n_patch)
    # print(img_shape)
    k = 0 #迭代所有的子块
    for i in range(N_full_imgs):
        for d in range(n_patch[0]):
            for h in range(n_patch[1]):
                for w in range(n_patch[2]):
                    full_prob[i,:,d*stride_shape[0]:(d*stride_shape[0])+patch_shape[0],
                                h*stride_shape[1]:(h*stride_shape[1])+patch_shape[1],
                                w*stride_shape[2]:(w*stride_shape[2])+patch_shape[2]]+=preds[k]
                    full_sum[i,:,d*stride_shape[0]:(d*stride_shape[0])+patch_shape[0],
                                h*stride_shape[1]:(h*stride_shape[1])+patch_shape[1],
                                w*stride_shape[2]:(w*stride_shape[2])+patch_shape[2]]+=1
                    # print(k)
                    k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0) 
    final_avg = full_prob/full_sum # 叠加概率 / 叠加权重 ： 采用了均值的方法
    # print (final_avg.shape)
    # assert(np.max(final_avg)<=1.0)
    # assert(np.min(final_avg)>=0.0)
    return final_avg

if __name__ == '__main__':
    a = np.random.randn(10)
    print(rm_max(a))
    full_imgs = np.ones(shape=(361, 431, 361))
    patch_shape = (64, 64, 64)
    stride_shape = (48, 48, 48)

    preds = []
    for pred in extract_ordered_overlap(full_imgs, patch_shape, stride_shape):
        preds.append([pred])
    final_img = recompone_overlap(preds, full_imgs.shape, stride_shape)