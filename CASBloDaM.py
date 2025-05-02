# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:20:51 2023

@author: SPAN_Karan

To save results of CASBloDaM method at  each stage by providiong origenal image and dl mask.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import blockage_calc_funcs
from skimage import morphology
l = [6,59,91,84,108]

for i in range(1,156):
    img_path = f"C:\\Users\\SPAN_Karan\\Desktop\\Karan\\Blockage_detection\\New dataset(full size)\\pre-processed mask\\{i} - Copy.bmp"
    mask_path = f"C:\\Users\\SPAN_Karan\\Desktop\\Karan\\Blockage_detection\\New dataset(full size)\\pre-processed mask\\{i}.bmp"
    result_path = "C:\\Users\\SPAN_Karan\\Desktop\\Karan\\Blockage_detection\\Results on each stage(new)\\"
    # result_path = "C:\\Users\\SPAN_Karan\\Desktop\\Karan\\Blockage_detection\\Results for paper\\"
    img = cv2.imread(img_path, 0)
    cv2.imwrite(result_path+f"{i}_orig_img.bmp", img)
    mask = cv2.imread(mask_path, 0)
    cv2.imwrite(result_path+f"{i}_dl_mask.bmp", mask)
    
    cath_point  = blockage_calc_funcs.find_cath_point(mask)
    
    mask_cath,std_width1,std_width2,cath_width_list1,cath_width_list2  = blockage_calc_funcs.find_cath_object(mask, cath_point, visulize=False)
    cv2.imwrite(result_path+f"{i}_cath_mask.bmp", mask_cath)
    
    w1 = sum(cath_width_list1)/len(cath_width_list1)
    w2 = sum(cath_width_list2)/len(cath_width_list2)
                                          
    plt.figure(figsize = (5,5))
    # plt.subplot(1,2,1)
    # plt.imshow(mask_cath,cmap = 'gray')
    # plt.title(f" Std1:{round(std_width1,3)}, Std2:{round(std_width2,3)}, W1:{round(w1,3)}, W2:{round(w2,3)}")
    # plt.subplot(1,2,2)
    plt.title(f" Std1:{round(std_width1,3)}, Std2:{round(std_width2,3)}, W1:{round(w1,3)}, W2:{round(w2,3)}")
    plt.plot(range(len(cath_width_list1)),cath_width_list1, label="width")
    # plt.plot(range(len(cath_width_list1)),cath_width_list2, label="w2")
    plt.legend()
    plt.savefig(result_path+f"{i}_cath_graph.png")
    
    skel_mask = morphology.thin(~mask,np.inf)
    
    skel = morphology.skeletonize(~mask)
    skel.astype(np.uint8)
    
    cv2.imwrite(result_path+f"{i}_thin_mask.bmp", skel_mask*255)
    
    # =========================Remove sub arteris==============================
    
    main_arteries,sub_arteries = blockage_calc_funcs.extrect_arteries(mask,
                                                                      skel_mask,
                                                                      w2,
                                                                      result_path,
                                                                      i)
    
    main_arteries_mask = blockage_calc_funcs.remove_subartery(mask,sub_arteries)
    mask = morphology.remove_small_holes(mask, 100)
    mask= mask*255
    cv2.imwrite(result_path+f"{i}_main_arteries_mask.bmp", mask)
    
    
    # ================ Width measurment & Blockage detection===================
    mask  = np.array(mask, np.uint8)
    mask_edge  = cv2.Canny(mask,50,205)
    
    regnew = blockage_calc_funcs.seg_obj_to_img(mask,
                                                skel_mask,
                                                main_arteries)
    
    # cv2.imshow("regnew",regnew)
    # cv2.waitKey(0)
    
    final_mask, no_blockages = blockage_calc_funcs.locate_blockage(main_arteries,
                                                      mask_edge,
                                                      mask,
                                                      regnew,
                                                      skel_mask,
                                                      0,
                                                      result_path,
                                                      i)
    
    # cv2.imshow("final_mask",final_mask)
    # cv2.waitKey(0)
    print(f"{i} Image is done.")
    
    
    
    
