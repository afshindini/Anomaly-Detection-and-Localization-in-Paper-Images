import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_recall_curve
from skimage import morphology 
from glob import glob
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from skimage.segmentation import mark_boundaries


from utils import read_img, get_patch, patch2img
from network import AutoEncoder
from options import Options


cfg = Options().parse()

# network
autoencoder = AutoEncoder(cfg)

if cfg.weight_file:
    autoencoder.load_weights(cfg.chechpoint_dir + '/' + cfg.weight_file)
else:
    file_list = os.listdir(cfg.chechpoint_dir)
    latest_epoch = max([int(i.split('-')[0]) for i in file_list if 'hdf5' in i])
    print('load latest weight file: ', latest_epoch)
    autoencoder.load_weights(glob(cfg.chechpoint_dir + '/' + str(latest_epoch) + '*.hdf5')[0])

autoencoder.summary()

def get_residual_map(img_path, cfg):
    test_img = read_img(img_path, cfg.grayscale)

    if test_img.shape[:2] != (cfg.im_resize, cfg.im_resize):
        test_img = cv2.resize(test_img, (cfg.im_resize, cfg.im_resize))
    if cfg.im_resize != cfg.mask_size:
        tmp = (cfg.im_resize - cfg.mask_size)//2
        test_img = test_img[tmp:tmp+cfg.mask_size, tmp:tmp+cfg.mask_size]

    original_img = test_img
    edges_filter = cv2.medianBlur(test_img,cfg.edge_filter)
    canny_img = cv2.Canny(edges_filter, 60, 80) / 255.

    test_img = cv2.medianBlur(test_img,cfg.filter)
    test_img_ = test_img / 255.

    if test_img.shape[:2] == (cfg.patch_size, cfg.patch_size):
        test_img_ = np.expand_dims(test_img_, 0)
        decoded_img = autoencoder.predict(test_img_)
    else:
        patches = get_patch(test_img_, cfg.patch_size, cfg.stride)
        patches = autoencoder.predict(patches)
        decoded_img = patch2img(patches, cfg.im_resize, cfg.patch_size, cfg.stride)

    rec_img = np.reshape((decoded_img * 255.).astype('uint8'), test_img.shape)

    if cfg.grayscale:
        ssim_residual_map = 1 - ssim(test_img, rec_img, win_size=cfg.ssim_win, full=True)[1]
        tot_ssim = 1 -  ssim(test_img, rec_img)
        l1_residual_map = np.abs(test_img / 255. - rec_img / 255.)
    else:
        ssim_residual_map = ssim(test_img, rec_img, win_size=cfg.ssim_win, full=True, multichannel=True)[1]
        ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)
        l1_residual_map = np.mean(np.abs(test_img / 255. - rec_img / 255.), axis=2)

    return test_img, rec_img, original_img, canny_img, ssim_residual_map, l1_residual_map, tot_ssim


def get_threshold(cfg):
    print('estimating threshold...')
    test_samples = glob(cfg.test_dir + '/**/*.jpg')
    print("test samples are", len(test_samples))
    print("first test sample is", test_samples[0])
    total_rec_ssim, total_rec_l1 = [], []
    total_tot_ssim = []
    gt = []
    for img_path in test_samples:
        _, _, _, _, ssim_residual_map, l1_residual_map, tot_ssim = get_residual_map(img_path, cfg)
        total_rec_ssim.append(ssim_residual_map)
        total_rec_l1.append(l1_residual_map)
        total_tot_ssim.append(tot_ssim)
        gt.append(0 if "good" in img_path else 1)

    total_rec_ssim = np.array(total_rec_ssim)
    total_rec_l1 = np.array(total_rec_l1)
    
    total_tot_ssim = np.array(total_tot_ssim)
    gt = np.array(gt)

    ssim_threshold = float(np.percentile(total_rec_ssim, [cfg.percent]))
    l1_threshold = float(np.percentile(total_rec_l1, [cfg.percent]))
    tot_ssim_threshold = float(np.percentile(total_tot_ssim, [cfg.percent]))



    fpr, tpr, _ = roc_curve(gt, total_tot_ssim)
    img_roc_auc = roc_auc_score(gt, total_tot_ssim)
    print('image ROCAUC with total ssim is: %.3f' % (img_roc_auc))
    plt.plot(fpr, tpr, label='ssim: %.3f' % (img_roc_auc))
    precision, recall, thresholds = precision_recall_curve(gt.flatten(), total_tot_ssim.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    f1ssimopt = thresholds[np.argmax(f1)]
    print('f1 optimal threshold ssim is: %.3f' % (f1ssimopt))


    plt.title('Image ROCAUC')
    plt.legend(loc="lower right")
    plt.savefig(cfg.save_dir+'/'+ 'roc_curve.png', dpi=100)
    plt.close()


    print('ssim_threshold: %f, l1_threshold: %f' %(ssim_threshold, l1_threshold))
    if not cfg.ssim_threshold:
        cfg.ssim_threshold = ssim_threshold
    if not cfg.l1_threshold:
        cfg.l1_threshold = l1_threshold
        
    return tot_ssim_threshold, f1ssimopt


def get_depressing_mask(cfg):
    depr_mask = np.ones((cfg.mask_size, cfg.mask_size)) * 0
    depr_mask[5:cfg.mask_size-5, 5:cfg.mask_size-5] = 1
    cfg.depr_mask = depr_mask


def get_results(file_list, cfg, tot_ssim_threshold, f1ssimopt):
    for img_path in file_list:
        img_name = img_path.split('/')[-1].split('.')[0]
        c = '' if not cfg.sub_folder else k
        test_img, rec_img, original_img, canny_img, ssim_residual_map, l1_residual_map, _ = get_residual_map(img_path, cfg)

        ssim_residual_map *= cfg.depr_mask
        if 'ssim' in cfg.loss:
            l1_residual_map *= cfg.depr_mask

        mask = np.zeros((cfg.mask_size, cfg.mask_size))
        mask[ssim_residual_map > cfg.ssim_threshold] = 1
        mask[l1_residual_map > cfg.l1_threshold] = 1

            

        kernel = morphology.disk(cfg.mor_filter)
        mask = morphology.opening(mask, kernel)
        
        if np.sum(mask*canny_img) < 7 :
            mask = np.zeros((cfg.mask_size, cfg.mask_size))
    
        mask *= 255
        vis_img = mark_boundaries(original_img, mask, color=(1, 0, 0), mode='thick')
        
        
        fig_img, ax_img = plt.subplots(1, 7, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(original_img, cmap='gray')
        ax_img[0].title.set_text('Original')
        ax_img[1].imshow(test_img, cmap='gray')
        ax_img[1].title.set_text('Filtered')
        ax_img[2].imshow(rec_img, cmap='gray')
        ax_img[2].title.set_text('Reconstructed')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Visualization')
        ax_img[5].imshow(test_img, cmap='gray', interpolation='none')
        ax_img[5].imshow(ssim_residual_map*255, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[5].title.set_text('Heat Map')
        ax_img[6].imshow(canny_img, cmap='gray')
        ax_img[6].title.set_text('Canny')

        fig_img.suptitle(f"pixels:{np.sum(mask*canny_img)}, unique: {np.unique(mask)}")
        fig_img.savefig(cfg.save_dir+'/'+c+'_'+img_name+'.png', dpi=100)
        plt.close()



if __name__ == '__main__':
    if not cfg.ssim_threshold or not cfg.l1_threshold:
       tot_ssim_threshold, f1ssimopt= get_threshold(cfg)

    get_depressing_mask(cfg)

    if cfg.sub_folder:
        for k in cfg.sub_folder:
            test_list = glob(cfg.test_dir+'/'+k+'/*')
            get_results(test_list, cfg, tot_ssim_threshold, f1ssimopt)
    else:
        test_list = glob(cfg.test_dir+'/*')
        print("test_list directory is", test_list)
        get_results(test_list, cfg, tot_ssim_threshold, f1ssimopt)
