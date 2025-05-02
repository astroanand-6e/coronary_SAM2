# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:10:00 2023

@author: SPAN_Karan
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.measure import label
from plantcv import plantcv as pcv             
from skimage import morphology
import scipy
from plantcv.plantcv import find_objects

__all__ = ['find_cath_point', 'find_cath_object', 'extrect_arteries','seg_obj_to_img',
           'find_nearest_btwn_angls', 'remove_subartery','locate_blockage', 
           'smooth','rotate','find_angles','angle_new','find_nearest_white',
           'arange_order_new','find_naighbors','intersect2d', 'max_area', 'rem_border']



def find_cath_point(mask):
    """
    Find the catheter point in a binary mask.

    Parameters:
    - mask (numpy.ndarray): A 2D binary mask where the catheter point needs to be found.

    Returns:
    - cath_point (tuple): A tuple containing the row and column indices of the catheter point.
                         If the cathode point is found in row 'r' and column 'c', the tuple is (r, c).

    Note:
    The function iterates through the rows and columns of the mask to find the first occurrence of
    a zero (0) value. It returns the indices of that occurrence as the cathode point.
    """
    for r_c in range(mask.shape[0]):
        if 0 in mask[r_c,:]:

            cath_point = (r_c ,np.min(np.where(mask[r_c,:]==0)))
            break
        elif 0 in mask[:,r_c]:

            cath_point = (np.min(np.where(mask[:,r_c]==0)) ,r_c)
            break
    return cath_point



def find_cath_object(mask, cath_point, visulize=False):
    """
   Find and characterize the catheter in a binary mask.

   Parameters:
   - mask (numpy.ndarray): A 2D binary mask representing the catheter object.
   - cath_point (tuple): A tuple containing the row and column indices of the catheter point.
   - visulize (bool, optional): If True, visualize intermediate steps of the process. Default is False.

   Returns:
   - mask (numpy.ndarray): The modified mask with the catheter object highlighted.
   - std_width1 (float): The standard deviation of catheter widths calculated from one side.
   - std_width2 (float): The standard deviation of catheter widths calculated from the other side.
   - cath_width_list1 (list): List of catheter widths calculated from one side for each point.
   - cath_width_list2 (list): List of catheter widths calculated from the other side for each point.

   Note:
   This function processes the input mask to thin it, prune small objects, and then characterize the cathode object
   by calculating its width at different points. It highlightes the catheter object from the original mask and returns
   the modified mask along with standard deviations and width lists.
   """
    mask = rem_border(mask)
    skel_mask = cv2.ximgproc.thinning(~mask)
    skel_mask = rem_border(skel_mask)
    
    _,_,segment_objects = pcv.morphology.prune(skel_img=skel_mask)

    std_width1=2
    std_width2=2
    
    while std_width1>1 or std_width2>1:
        dist = np.inf
        for idx ,segment_object in enumerate(segment_objects):
            # assumption : if length of object is <35 then it is a noice
            if len(segment_object)>35:
                segment_object = np.squeeze(segment_object, axis = 1)    
                segment_object = arange_order_new(segment_object)
                
                end_point1 = segment_object[0]
                end_point2 = segment_object[-1]
                new_dist  = min(math.dist(end_point1,np.flip(cath_point)),
                            math.dist(end_point2,np.flip(cath_point)))
                
                # to find object which is nearest to the cath_point 
                if new_dist<dist:
                    dist = new_dist
                    cath_idx = idx
                    
                # if visulize:
                #     print(new_dist, len(segment_object))
                #     image = skel_mask.copy()
                #     image = cv2.circle(image, end_point1, 3, 200, -1)
                #     image = cv2.circle(image, end_point2, 3, 200, -1)
                    
                #     cv2.imshow("Image",image)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                    
        segment_object = segment_objects[cath_idx]
        del segment_objects[cath_idx]
        segment_object = np.squeeze(segment_object, axis = 1)    
        segment_object = arange_order_new(segment_object)
                
        length = len(segment_object)
        
        cath_width_list1 = []
        cath_width_list2= []
        mask_edge = cv2.Canny(mask,50,205)
        for point in range(length//5,length-(length//5)): # can be updated
     
            point1 = ((segment_object[np.clip(point-3,0,512),0]),(segment_object[np.clip(point-3,0,512),1]))
            point2 = (segment_object[point+3,0],
                      (segment_object[point+3,1]))
            P = np.flip(segment_object[point]) 
    
            ang = angle_new(np.flip(point1), np.flip(point2))
            
            a1 = (180 - (90/2))/57.29
            a2 = (90/2)/57.29
            
            ang1 = ang + a1 - 6.28 if (ang + a1)>6.28 else ang + a1
            ang2 = ang + a2 -6.28 if (ang + a2)>6.28 else ang + a2
            
            P1, w1, flag = find_nearest_btwn_angls(mask_edge, P, ang1, ang2,visulize)
            
            if flag:
                
                ang2 = ang -a1 +6.28 if (ang - a1)<0  else ang -a1
                ang1 = ang - a2 +6.28 if (ang - a2)<0  else ang - a2
                
                P2, w2, flag = find_nearest_btwn_angls(mask_edge, P, ang1, ang2, visulize)
           
            if flag:
    
                # print("Width :",round(math.dist(P1,P2),2))
                cath_width_list1.append(round(w1+w2,2)) 
                cath_width_list2.append(round(math.dist(P1,P2),2)) 
                
            # to visulize point vise width       
            if visulize:
                viz_img = mask.copy()
                P = np.flip(segment_object[point])
                viz_img[P[0], P[1]] = ~viz_img[P[0], P[1]]
                # viz_img[P1[1], P1[0]] = ~viz_img[P1[1], P1[0]]
                # viz_img[P2[1], P2[0]] = ~viz_img[P2[1], P2[0]]
                cv2.line(viz_img,segment_object[point],P1,100,1)
                cv2.line(viz_img,segment_object[point],P2,100,1)
                cv2.line(viz_img,P2,P1,255,1)
                cv2.namedWindow('Image', cv2.WINDOW_KEEPRATIO)
                cv2.imshow("Image",viz_img)
                # cv2.resizeWindow('Image', 1000, 1000)
                cv2.waitKey(0)
        cv2.destroyAllWindows()
                
        
        std_width1  = np.std(cath_width_list1)
        std_width2  = np.std(cath_width_list2)
        # print(std_width)
        
    
    segment_object = segment_object[length//5:length-(length//5),:]                 
    for idx,_ in enumerate(segment_object):
        
        mask[segment_object[idx,1],segment_object[idx,0]] = ~mask[segment_object[idx,1],
                                                                  segment_object[idx,0]] 
    
    return mask,std_width1,std_width2,cath_width_list1,cath_width_list2

def rem_border(image, border_value = 0):
    """
    Remove the border of a 2D image by setting border pixels to a specified value.

    Parameters:
    - image (numpy.ndarray): Input 2D image from which the border needs to be removed.
    - border_value (int, optional): Value to be assigned to the border pixels. Default is 0.

    Returns:
    - img (numpy.ndarray): A copy of the input image with the border pixels set to the specified value.
    """
    img = image.copy()
    img[0,:]=0
    img[img.shape[0]-1,:]=0
    img[:,0]=0
    img[:,img.shape[1]-1]=0
    return img

def max_area(out_img):
  """
    Find and extract the region with the maximum area in a labeled image.

    Parameters:
    - out_img (numpy.ndarray): Input labeled image where regions are identified.

    Returns:
    - reg1 (numpy.ndarray): Binary image containing the region with the maximum area.

    Note:
    This function takes a labeled image and identifies the region with the maximum area.
    It returns a binary image containing only the pixels belonging to the identified region.
  """
  label_img = label(out_img)

  area =[]
  for prop in range(len(np.unique(label_img))):
    white_coords = np.argwhere(label_img == prop)
    area.append(len(white_coords))

  for i in range(len(area)):
    if area[i]==np.max(area):
      break
  area[i]=0

  for index in range(len(area)):
    if area[index]==np.max(area):
      break

  white_coords = np.argwhere(label_img == index)

  reg1 = np.zeros((out_img.shape))

  for i in range(len(white_coords)):
    # print(white_coords[i])
    reg1[white_coords[i,0],white_coords[i,1]]=255

  return np.array(reg1, dtype="uint8")

# def get_cathator(out_img,point):

#   label_img = label(~out_img)

#   # calculate are af each part
#   # area =[]
#   for prop in range(len(np.unique(label_img))):
#     white_coords = np.argwhere(label_img == prop)
    
#     if np.any(np.all(white_coords == point, axis=1)):
#         break
    
#   white_coords = np.argwhere(label_img == prop)

#   reg1 = np.zeros((out_img.shape))

#   for i in range(len(white_coords)):
#     # print(white_coords[i])
#     reg1[white_coords[i,0],white_coords[i,1]]=255

#   return np.array(reg1, dtype="uint8")

def intersect2d(A,B):
    """
    Find the intersection of two 2D numpy arrays along the first axis.

    Parameters:
    - A (numpy.ndarray): First input 2D array.
    - B (numpy.ndarray): Second input 2D array.

    Returns:
    - C (numpy.ndarray): 2D array containing the common elements between A and B along the first axis.

    Note:
    This function takes two 2D arrays, A and B, and returns a new array C containing the common elements
    along the first axis. The data types of A and B should be the same. The resulting array is not sorted
    unless the input arrays are sorted.

    Reference:
    - np.intersect1d: https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html
    ```
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    ```
    """
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}
    
    C = np.intersect1d(A.view(dtype), B.view(dtype))
    
    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C

def find_naighbors(point,segment_object):
    """
    Find the neighboring points around a given point in a 2D segment object.

    Parameters:
    - point (tuple): A tuple containing the (x, y) coordinates of the reference point.
    - segment_object (numpy.ndarray): A 2D array representing a segment object.

    Returns:
    - neighbors (numpy.ndarray): An array containing the (x, y) coordinates of the neighboring points.

    Note:
    This function takes a reference point (x, y) and a 2D segment object. It returns an array containing the
    (x, y) coordinates of the neighboring points around the reference point in the segment object.
    """
    
    x,y = point
    return np.array(((x-1,y-1),(x-1,y),(x-1,y+1),
                     (x,y-1),(x,y+1),
                     (x+1,y-1),(x+1,y+1),(x+1,y)), 
                    dtype="int32",like = segment_object)

# def arange_order(segment_object):
#     """
#    Arrange the points in a 2D segment object to ensure a consistent order.

#    Parameters:
#    - segment_object (numpy.ndarray): A 2D array representing a segment object.

#    Returns:
#    - segment_object (numpy.ndarray): The modified segment object with arranged points.

#    Note:
#    This function takes a 2D segment object and arranges its points to ensure a consistent order.
#    The order is determined based on the connectivity of points in the segment object. If the first point
#    is connected to multiple neighbors, the function rearranges the points until a consistent order is achieved.
#    """
#     neighbors = find_naighbors(segment_object[0],segment_object)
    
#     if len(intersect2d(segment_object,neighbors))==1:
#         return segment_object
#     else:
#         for ixd, point in enumerate(segment_object):

#             neighbors = find_naighbors(point,segment_object)
           
#             if len(intersect2d(segment_object,neighbors))==1:
#                 break
            
#         segment_object = np.concatenate((segment_object[ixd:],
#                                          segment_object[0:ixd]))
#         return segment_object
    
def arange_order_new(segment_object):
    """
   Arrange the unique points in a 2D segment object to ensure a consistent order.

   Parameters:
   - segment_object (numpy.ndarray): A 2D array representing a segment object.

   Returns:
   - arranged_segment_object (numpy.ndarray): The modified segment object with arranged points.

   Note:
   This function takes a 2D segment object and arranges its unique points to ensure a consistent order.
   The order is determined based on the connectivity of points in the segment object. If a point is connected
   to multiple neighbors, the function removes the point and continues until a consistent order is achieved.
   """
    segment_object = np.unique(segment_object, axis =0)
    neighbors = find_naighbors(segment_object[0],segment_object)
    arranged_segment_object = []
    for idx, point in enumerate(segment_object):

        neighbors = find_naighbors(point,segment_object)
       
        if len(intersect2d(segment_object,neighbors))==1:
            segment_object = np.delete(segment_object, idx,axis=0)
            break
        
    arranged_segment_object.append(point)
    for i in range(len(segment_object)-1):
        nonzero = segment_object
        target = point
        distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
        nearest_index = np.argmin(distances)
        point = nonzero[nearest_index]
        arranged_segment_object.append(point)
        segment_object = np.delete(segment_object,np.where(np.bitwise_and(segment_object[:,0] == point[0],segment_object[:,1] == point[1])),axis=0)
        # distances[nearest_index]

    return np.asanyarray(arranged_segment_object)
       
def extrect_arteries(mask, skel_mask,cath_width, result_path = None, img_num = None):
    
    """
    Extract main and sub-arteries from a binary mask based on skeletonized mask and catheter width.

    Parameters:
    - mask (numpy.ndarray): Binary mask representing the arteries.
    - skel_mask (numpy.ndarray): Skeletonized mask corresponding to the input mask.
    - cath_width (int): Catheter width used for classifying main and sub-arteries.
    - result_path (str, optional): Path to save the result images. Default is None.
    - img_num (str, optional): Image number or identifier for saving result images. Default is None.

    Returns:
    - main_arteries (list): List of main arteries represented as arrays of (x, y) coordinates.
    - sub_arteries (list): List of sub-arteries represented as arrays of (x, y) coordinates.

    Note:
    This function takes a binary mask representing arteries, its corresponding skeletonized mask, and the cathode width.
    It classifies arteries into main and sub-arteries based on their width. The result can be saved as images if a result path
    and image number are provided.
    """
    skel_mask = skel_mask.astype(np.uint8)
    
    # pcv.params.debug = "plot"
    _, extrected_arteries, segment_objects = pcv.morphology.prune(skel_img=skel_mask)
    cv2.imwrite(result_path+f"{img_num}_extrected_arteries.bmp", extrected_arteries)
    
    sub_arteries = []
    main_arteries = []
    for segment_object in segment_objects:
        
        # from (no_of_objects,1,2) to (no_of_objects,2)
        segment_object = np.squeeze(segment_object, axis = 1)
        reg1 = np.zeros((skel_mask.shape),dtype="bool",like = skel_mask)
        obj_len = len(segment_object)
        
        # neighbors = find_naighbors(segment_object[0],segment_object)
        
        # if len(intersect2d(segment_object,neighbors))!=1:
  
        #     for ixd, point in enumerate(segment_object):

        #         neighbors = find_naighbors(point,segment_object)
               
        #         if len(intersect2d(segment_object,neighbors))==1:
        #             break
                
        #     segment_object = np.concatenate((segment_object[ixd:],
        #                                      segment_object[0:ixd]))
        segment_object = arange_order_new(segment_object)
        
        for i in range(len(segment_object)//2):
            reg1[segment_object[i,1],segment_object[i,0]]=True
            # fig = plt.figure(figsize=(20,10))
            # fig.subplots_adjust(hspace=1,wspace = 0.2)
            
            # fig.subplots_adjust(top = 1.15)
            # ax = fig.add_subplot(1, 2, 1)
            # ax.imshow(~mask,cmap = "gray")
            
            # ax = fig.add_subplot(1, 2, 2)
            # ax.imshow(reg1, cmap = "gray")
            
        # wather artery is main or sub
        """ ---------------Logic 1-------------"""
        # count = 0
        # area  = 0
        # width = 0
        # kernel = np.ones((3,3),np.uint8)
        # while (area<obj_len and width<cath_width):
        #   reg1 = np.array(reg1, dtype="uint8")
        #   reg1 = cv2.dilate(reg1,kernel,iterations = 1)
        
        #   # plt.figure()
        #   # plt.imshow(cv2.bitwise_and(reg1, mask, mask = None), cmap = 'gray')
        #   count+=1
        #   width = count*4-1
        #   area = np.sum(cv2.bitwise_and(reg1, mask, mask = None))
        
        
        """ ---------------Logic 2-------------"""
        kernel = np.ones((3,3),np.uint8) 
        old_area = np.inf
        new_area = 512*512
        count = -1
        while (old_area>new_area):
          old_area = new_area
          reg1 = np.array(reg1, dtype="uint8")
          reg1 = cv2.dilate(reg1,kernel,iterations = 1)
          diff = cv2.bitwise_xor(reg1, mask, mask=None)
          diff[(diff < 255) & (diff>5)] = 0
          diff[diff==1]=255
          # plt.figure()
          # plt.imshow(diff,cmap='gray')
          # print(np.sum(diff)//255)
          new_area = np.sum(~diff)//255
          count+=1
          
        art_width = count*4-1
        
        # if count>cath_count :
        if art_width < int(cath_width):
            sub_arteries.append(segment_object)
        else:
            main_arteries.append(segment_object)
    # segment_objects = np.delete(segment_objects,rem_ind)
    
    return main_arteries,sub_arteries

def seg_obj_to_img(mask,skel_mask, segment_objects,draw=False):
    """
    Convert segment objects to an image and optionally visualize the process.
    
    Parameters:
    - mask (numpy.ndarray): Binary mask representing the arteries.
    - skel_mask (numpy.ndarray): Skeletonized mask corresponding to the input mask.
    - segment_objects (list): List of segment objects represented as arrays of (x, y) coordinates.
    - draw (bool, optional): If True, visualize the process. Default is False.
    
    Returns:
    - regnew (numpy.ndarray): Binary image containing the segment objects.
    
    Note:
    This function converts a list of segment objects to a binary image. Each segment object is represented as an array
    of (x, y) coordinates. The resulting image is a combination of all segment objects. Visualization is optional.
    """
    
    regnew = np.zeros((mask.shape))
    reg1 = np.zeros((mask.shape))
    for k in range(len(segment_objects)):
      segment_object = np.asarray(segment_objects[k])
    
      # obj_len = len(segment_object)
      for i in range(len(segment_object)):
    
        regnew[segment_object[i,1],segment_object[i,0]]=255
    
      reg1 = cv2.bitwise_or(reg1,regnew, mask = None)
      # plt.imshow(reg1)
      # break
    if draw:
        fig = plt.figure(figsize=(30,10))
        fig.subplots_adjust(hspace=2,wspace = 0.2)
        
        fig.subplots_adjust(top = 1.15)
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(~mask+skel_mask,cmap = "gray")
        
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(skel_mask, cmap = "gray")
        
        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(regnew, cmap = "gray")
    return regnew

def find_nearest_white(img, target):
    """
    Find the nearest white pixel in a binary image to a given target point.
    
    Parameters:
    - img (numpy.ndarray): Binary image.
    - target (tuple): A tuple containing the (row, column) coordinates of the target point.
    
    Returns:
    - nearest_point (tuple): A tuple containing the (row, column) coordinates of the nearest white pixel.
    - distance (float): The Euclidean distance between the target point and the nearest white pixel.   
    """
    nonzero = np.argwhere(img == 255)
    distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index],distances[nearest_index]

# def angle(p1,p2):
#     # Difference in x coordinates
#     dx = p2[0] - p1[0]
    
#     # Difference in y coordinates
#     dy = p2[1] - p1[1]
    
#     # Angle between p1 and p2 in radians
#     theta = math.atan2(dy, dx)
#     if theta>=0:
#         return theta
#     return 6.28+theta

def angle_new(p1, p2):
    """
    Calculate the angle between two points in radians.

    Parameters:
    - p1 (tuple): A tuple containing the (x, y) coordinates of the first point.
    - p2 (tuple): A tuple containing the (x, y) coordinates of the second point.

    Returns:
    - theta (float): The angle between the two points in radians.

    """
    # Difference in x coordinates
    dx = p2[0] - p1[0]

    # Difference in y coordinates
    dy = p2[1] - p1[1]

    # Angle between p1 and p2 in radians
    theta = math.atan2(dy, dx)
    theta -= 3.14 / 2  # Adjust the angle to the desired orientation

    if theta >= 0:
        return theta
    return 6.28 + theta  # Ensure the result is in the range [0, 2*pi)


def find_angles(img, target):
    """
   Find the angles between a target point and all white pixels in a binary image.

   Parameters:
   - img (numpy.ndarray): Binary image.
   - target (tuple): A tuple containing the (row, column) coordinates of the target point.

   Returns:
   - nonzero (numpy.ndarray): Array containing the (row, column) coordinates of all white pixels.
   - angles (list): List of angles between the target point and each white pixel in radians.

   """
    nonzero = np.argwhere(img == 255)
    angles = [angle_new(target,i ) for i in nonzero]
    return nonzero,angles

def find_nearest_btwn_angls(edge_crop, target, ang1, ang2, visulize =False):
    """
    Find the nearest white pixel within specified angle range relative to a target point in a binary image.

    Parameters:
    - edge_crop (numpy.ndarray): Binary image.
    - target (tuple): A tuple containing the (row, column) coordinates of the target point.
    - ang1 (float): Starting angle of the range in radians.
    - ang2 (float): Ending angle of the range in radians.
    - visulize (bool, optional): If True, visualize the process. Default is False.
    
    Returns:
    - P (tuple): A tuple containing the (row, column) coordinates of the nearest white pixel.
    - w (float): The Euclidean distance between the target point and the nearest white pixel.
    - flag (bool): True if a white pixel is found within the specified angle range, False otherwise.

    Note:
    This function takes a binary image, a target point, and a specified angle range. It finds the nearest white pixel
    within the specified angle range relative to the target point and returns its coordinates, distance, and a flag
    indicating whether a white pixel was found.

    """
    # edge_crop[target[1],target[0]]=50
    nonzero,angles = find_angles(edge_crop, target)
    distances2=[]
    nonzero2 = []
    angles2 = []
    img = edge_crop.copy()
    if (ang1<=90/57.29 or ang2>=270/57.29):
        for idx, angle_p in enumerate(angles):
            if angle_p<ang1 or angle_p>ang2:
                img[nonzero[idx][0],nonzero[idx][1]]=100
                nonzero2.append(nonzero[idx])
                distances2.append(math.dist(nonzero[idx], target))
        if visulize == True:
            # img = edge_crop.copy()
            # img[nonzero[idx][0],nonzero[idx][1]] = ~img[nonzero[idx][0],nonzero[idx][1]]
            img[target[0],target[1]] = ~img[target[0],target[1]]
            
            # print("Target:", target,"artery:",np.flip(nonzero[idx]),angles[idx]*57.29)
            cv2.imshow("image1",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        img = edge_crop.copy()
        
        for idx, angle_p in enumerate(angles):
            # print(angle_p*57.29,(nonzero[idx][0],nonzero[idx][1]))
           
            # cv2.imshow("crop",cv2.resize(edge_crop,(512,512)))
            # cv2.waitKey(0)
            
            if angle_p<ang1 and angle_p>ang2:
                img[nonzero[idx][0],nonzero[idx][1]]=100
                nonzero2.append(nonzero[idx])
                angles2.append(angles[idx])
                distances2.append(math.dist(nonzero[idx],target))
            
        if visulize == True:
            # img = edge_crop.copy()
            # img[nonzero[idx][0],nonzero[idx][1]] = ~img[nonzero[idx][0],nonzero[idx][1]]
            img[target[0],target[1]] = ~img[target[0],target[1]]
            
            # print("Target:", target,"artery:",np.flip(nonzero[idx]),angles[idx]*57.29)
            cv2.imshow("image2",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # distances2=0
    if len(distances2)!=0:
        w = np.min(distances2)
        P =np.flip(nonzero2[np.argmin(distances2)])
        return P,w,True
    else:
        return 0,0,False
    
# def find_artery_width(edge_crop,target):
    
#     P1, w1 = find_nearest_white(edge_crop,target)
    
#     ang = angle_new(target ,P1)
#     ang1 = ang - 2.35 +6.28 if (ang - 2.35)<0  else ang - 2.35
#     ang2 = ang + 2.35 -6.28 if (ang + 2.35)>6.28  else ang + 2.35
     
#     nonzero,angles = find_angles(edge_crop, target)
#     distances=[]
#     nonzero2 = []
    
#     if (ang1<abs(ang1-ang2) or ang2>(6.28-abs(ang1-ang2))):
#         for idx, angle_p in enumerate(angles):
#             if angle_p<ang1 or angle_p>ang2:
#                 # edge_crop[nonzero[idx][0],nonzero[idx][1]]=50
#                 nonzero2.append(nonzero[idx])
#                 distances.append(math.dist(nonzero[idx], target))
                
#     else:
#         for idx, angle_p in enumerate(angles):
#             if angle_p<ang1 and angle_p>ang2:
#                 # edge_crop[nonzero[idx][0],nonzero[idx][1]]=50
#                 nonzero2.append(nonzero[idx])
#                 distances.append(math.dist(nonzero[idx],target))
#     # distances2=0
#     if len(distances)!=0:
#         w2 = np.min(distances)
#         P2 =np.flip(nonzero2[np.argmin(distances)])
#         P1 = np.flip(P1)
#         return P1,w1, P2,w2,True
#     else:
#         return P1,w1,0,0,False

# def find_cath(mask):
#     i, j, count =0, 0, 0
#     col_p  = False
#     while mask[i,j]!=0 and mask[j,i]!=0:
#       i = count//mask.shape[0]
#       j = count% mask.shape[1]
#       count+=1
#       if mask[j,i] == 0:
#         col_p  = True
      
#     # print(f"Cathator found at:{i,j}")
    
#     j = np.clip(j,64,512-64)
#     if col_p:
#       j+=10
#       croped = mask[j-64:j+64,i:i+128]
#     else:
#       i+=10
#       croped  = mask[i:i+128,j-64:j+64]
#     # plt.imshow(croped,cmap='gray')
#     croped = max_area(~croped)
#     # plt.imshow(croped,cmap='gray')
    
#     return i,j,croped

# def cathator_width(croped):
#     skel_croped = morphology.thin(croped,np.inf)
#     # plt.imshow(skel_croped,cmap='gray')
#     skel_len = np.sum(skel_croped)
    
#     kernel = np.ones((3,3),np.uint8) 
#     area = 0
#     cath_count = 0
#     while (area<skel_len*2):
#       skel_croped = np.array(skel_croped, dtype="uint8")
#       skel_croped = cv2.dilate(skel_croped,kernel,iterations = 1)
#        # plt.figure()
#        # plt.imshow(cv2.bitwise_and(skel_croped, ~croped, mask=None),cmap='gray')
#       cath_count+=1
#       area = np.sum(cv2.bitwise_and(skel_croped, ~croped, mask = None))
    
#     cat_width = 4*(cath_count-1) -1
#     return cat_width,cath_count

def remove_subartery(mask,sub_arteries):
    
    """
    Remove sub-arteries from a binary mask based on their coordinates.
    
    Parameters:
    - mask (numpy.ndarray): Binary mask representing the arteries.
    - sub_arteries (list): List of sub-arteries represented as arrays of (x, y) coordinates.
    
    Returns:
    - modified_mask (numpy.ndarray): Binary mask with sub-arteries removed.
    
    Note:
    This function takes a binary mask representing arteries and a list of sub-arteries represented as arrays of (x, y) coordinates.
    It removes the sub-arteries from the binary mask by connecting the endpoints of each sub-artery and updating the mask accordingly.
    
    """

    mask_edge  = cv2.Canny(mask,50,205)
    # plt.imshow(mask_edge, cmap='gray')
    # mask1 = mask.copy()
    for sub_artery in sub_arteries:
        # if len(sub_artery)>5:
        for i in range(2,len(sub_artery)//2 -3):
            point1 = np.flip(sub_artery[i])
            point2 = np.flip(sub_artery[i+4])
            # target = (32+point1[1]F-x, 32+point1[0]-y)
            # target2 = (target[0]+diff[1], target[1]+diff[0])
            
            ang = angle_new(point1, point2)
            
            # ang1 = ang + 2.35 -6.28 if (ang + 2.35)>6.28 else ang + 2.35
            # ang2 = ang + 0.78 -6.28 if (ang + 0.78)>6.28 else ang + 0.78
            a1 = (180 - (90/2))/57.29
            a2 = (90/2)/57.29
            
            ang1 = ang + a1 - 6.28 if (ang + a1)>6.28 else ang + a1
            ang2 = ang + a2 -6.28 if (ang + a2)>6.28 else ang + a2
            
            P1, w1, flag = find_nearest_btwn_angls(mask_edge, point1, ang1, ang2)
            
            if flag:
                
                # ang2 = ang - 2.35 +6.28 if (ang - 2.35)<0  else ang - 2.35
                # ang1 = ang - 0.78 +6.28 if (ang - 0.78)<0  else ang - 0.78
                ang2 = ang -a1 +6.28 if (ang - a1)<0  else ang -a1
                ang1 = ang - a2 +6.28 if (ang - a2)<0  else ang - a2
                
                P2, w2, flag = find_nearest_btwn_angls(mask_edge, point1, ang1, ang2)

            if flag:
                mask = cv2.line(mask, P1, P2, (255), 3)
            # plt.figure(figsize=(10,10))
        # plt.imshow(mask, cmap='gray')
    return mask

def smooth(y):
    """
    Smooth a 1D signal using a simple moving average filter.

    Parameters:
    - y (numpy.ndarray): 1D array representing the signal.

    Returns:
    - y_smooth (numpy.ndarray): Smoothed 1D array.

    """
    box = np.asarray([16,9,4,1,4,9,16])/55
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    Parameters:
    - origin (tuple): A tuple containing the (x, y) coordinates of the rotation origin.
    - point (tuple): A tuple containing the (x, y) coordinates of the point to be rotated.
    - angle (float): The angle of rotation in radians.

    Returns:
    - rotated_point (tuple): A tuple containing the (x, y) coordinates of the rotated point.

    Note:
    This function rotates a point counterclockwise by a given angle around a specified origin.
    The angle should be given in radians.

    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)

def locate_blockage(main_arteries,mask_edge,mask,regnew,skel_mask,Flag = 0,result_path = None, img_num = None):
    """
    Locate blockages in main arteries based on width analysis.

    Parameters:
    - main_arteries (list of arrays): List of main arteries, each represented as an array of points.
    - mask_edge (numpy array): Canny edge-detected version of the original mask.
    - mask (numpy array): Original mask representing the blood vessels.
    - regnew (numpy array): Empty array to store the segmented regions.
    - skel_mask (numpy array): Thinned version of the mask.
    - Flag (int): Flag to control the visualization. Default is 0.
    - result_path (str): Path to save the results. Default is None.
    - img_num (str): Image number for saving results. Default is None.

    Returns:
    - mask (numpy array): Updated mask with blockages marked.
    - block_num (int): Number of detected blockages.

    Note:
    This function locates blockages in main arteries based on width analysis and saves the results.

    """
    block_num = 0
    image2 = mask.copy()
    # Iterate over each main artery
    for obj_idx,main_artery in enumerate(main_arteries):
        
        artery_width = []
        artery_width1 = []
        # Iterate over each point in the main artery
        for point in range(len(main_artery)//2):
            
            #  Define points for width analysis
            point1 = (main_artery[point,0]-32,(main_artery[point,1]-32))
            point2 = (main_artery[np.clip(point+4,0,len(main_artery)-1),0]-32,
                      (main_artery[np.clip(point+4,0,len(main_artery)-1),1]-32))
            # Crop the region for width analysis
            diff = (point2[0]-point1[0], point2[1]-point1[1])
            y,x = np.clip(point1,0,512-64)
            
            edge_crop = mask_edge[x:x+64,y:y+64].copy()
            target = (32+point1[1]-x, 32+point1[0]-y)
            target2 = (target[0]+diff[1], target[1]+diff[0])
            
            # Set angles for width analysis
            ang = angle_new(target, target2)
            
            # ang1 = ang + 2.35 -6.28 if (ang + 2.35)>6.28 else ang + 2.35
            # ang2 = ang + 0.78 -6.28 if (ang + 0.78)>6.28 else ang + 0.78
            a1 = (180 - (90/2))/57.29
            a2 = (90/2)/57.29
            
            ang1 = ang + a1 - 6.28 if (ang + a1)>6.28 else ang + a1
            ang2 = ang + a2 -6.28 if (ang + a2)>6.28 else ang + a2
            
            P1, w1, flag = find_nearest_btwn_angls(edge_crop, target, ang1, ang2)
            
            if flag:
                
                # ang2 = ang - 2.35 +6.28 if (ang - 2.35)<0  else ang - 2.35
                # ang1 = ang - 0.78 +6.28 if (ang - 0.78)<0  else ang - 0.78
                
                ang2 = ang -a1 +6.28 if (ang - a1)<0  else ang -a1
                ang1 = ang - a2 +6.28 if (ang - a2)<0  else ang - a2
                
                P2, w2, flag = find_nearest_btwn_angls(edge_crop, target, ang1, ang2)
           
            if flag:
    
                artery_width1.append(round(math.dist(P1,P2),2)) 
                artery_width.append(round(w1+w2,2)) 
                
                # Visualization of width analysis
                edge_crop1= mask_edge[x:x+64,y:y+64].copy()
                edge_crop1 = (cv2.line(edge_crop1, P1, P2, (100), 1))
                edge_crop1[target[0],target[1]]=255
                edge_crop1[target2[0],target2[1]]=255
                edge_crop1[0:20,0:40]=np.zeros((20,40))
                edge_crop1 = cv2.putText(edge_crop1, str(round(math.dist(P1,P2),2)),(0,10),cv2.FONT_HERSHEY_COMPLEX,0.4, 255, 1, cv2.LINE_4)
                edge_crop2 = cv2.line(edge_crop, P1, (target[1],target[0]), (100), 1)
                edge_crop2 = cv2.line(edge_crop2, (target[1],target[0]), P2, (200), 1)
                edge_crop2[0:20,0:40]=np.zeros((20,40))
                edge_crop2 = cv2.putText(edge_crop2, str(round((w1+w2),2)),(0,10),cv2.FONT_HERSHEY_COMPLEX,0.4, 255, 1, cv2.LINE_4)
                    
                mask_rect = (mask+regnew).copy()
                mask_rect = cv2.rectangle(mask_rect, (y,x), (y+64,x+64), 0, 1)
                cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Mask", 500, 500)
                cv2.imshow("Mask",mask_rect)
                
                test_width = cv2.hconcat([edge_crop1,edge_crop2])
                cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Resized_Window", 600, 300)
                cv2.imshow("Resized_Window",test_width)  
                if Flag !=0:
                    cv2.waitKey(0)
        # ===============================================================================
        # example data with peaks:
        # data = np.asarray(smooth(data,filt_size))
        if len(artery_width)>10:
            artery_width = np.asarray(artery_width)
            data = np.round_(scipy.signal.wiener(artery_width,10),decimals = 1)
            data = np.asarray(smooth(data))
            shift = (len(artery_width)-len(data))//2
            x = range(shift,len(data)+shift)
            
            #     ___ detection of local minimums and maximums ___
            
            # a = np.diff(np.sign(np.diff(data))).nonzero()[0] + 1            # local min & max
            b = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1 + shift # local min
            c = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1 + shift  # local max
            # +1 due to the fact that diff reduces the original index number
            # ===================================================================================
           
            # Plot main graph.
            # (fig, ax) = plt.subplots(ncols=2)
            # fig.set_size_inches(20, 10)
            regnew = seg_obj_to_img(mask,skel_mask,np.expand_dims(main_artery,axis=0),draw=False)
            
            # ax[1].plot(x, data)
            # Plot peaks.
            peak_x = np.append([0],c)
            peak_y = artery_width[peak_x]
            # ax[1].plot(peak_x, peak_y, marker='o',linestyle='None', color='green', label="Peaks")
             
            # Plot valleys.
            valley_x = b
            valley_y = artery_width[b]
            # ax[1].plot(valley_x, valley_y, marker='o',linestyle='None', color='red', label="Valleys")
            # ax[1].plot(range(len(artery_width)), artery_width)
            
        # =================================filter the vallies========================================

            for idx,valley in enumerate(valley_y):
                r_max = max(artery_width[0:valley_x[idx]])
                l_max = max(artery_width[valley_x[idx]:])
                
                if r_max-valley>4 and l_max-valley>4:
                    norm_width = (r_max+r_max)/2
                    Blockage = int((1-(valley/norm_width)**2)*100)                    
                
                    if Blockage>=60:
                    
                        block_num+=1
                        # cv2.imshow("regnew",regnew + mask)
                        # cv2.waitKey(0)
                        cv2.imwrite(result_path+f"{img_num}_block_artery_{block_num}.bmp", regnew + mask)
                        
                        plt.figure(figsize = (5,5))
                        # plt.subplot(1,2,1)
                        # plt.imshow(mask,cmap = 'gray')
                        # plt.title(f"Image:{path+1}, Std1:{round(std_width1,3)}, Std2:{round(std_width2,3)}, W1:{round(w1,3)}, W2:{round(w2,3)}")
                        # plt.subplot(1,2,2)
                        plt.plot(x, data, label="width")
                        # plt.plot(range(len(artery_width)),artery_width, label="w1")
                        # plt.plot(range(len(artery_width1)),artery_width1, label="w2")
                        plt.legend()
                        plt.savefig(result_path+f"{img_num}_width_graph_{block_num}_cath_graph.png")
                        
                        org1  = (main_artery[valley_x[idx],0],(main_artery[valley_x[idx],1]))
                        org2   = (main_artery[np.clip(valley_x[idx]+30,0,len(main_artery)-1),0],
                                 main_artery[np.clip(valley_x[idx]+30,0,len(main_artery)-1),1])
                        
                        (x1,y1) = rotate(org1, org2, -90/57)
                        
                        image = mask.copy()
                        
                        mask_b = cv2.putText(image, str(Blockage)+"%", 
                                           (x1,y1), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.7, 100, 2, cv2.LINE_AA)
                        mask_b = cv2.circle(mask_b, org1, 10, 100, 2)
                        
                        image2 = cv2.putText(image2, str(Blockage)+"%", 
                                           (x1,y1), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.7, 100, 2, cv2.LINE_AA)
                        image2 = cv2.circle(image2, org1, 10, 100, 2)
                        
                        cv2.imwrite(result_path+f"{img_num}_blockage_{block_num}.bmp", mask_b)
            
        # ====================================================================================================
            # plt.subplot(1,2,1)
            # plt.axis('off')
            # plt.imshow(regnew+mask,cmap = "gray") 
        
    cv2.destroyAllWindows()
    cv2.imwrite(result_path+f"{img_num}_All_blockage.bmp", image2)
    return mask, block_num