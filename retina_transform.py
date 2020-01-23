import cv2
import numpy as np
import sys
import os
import time

def genGaussiankernel(width, sigma):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d

def pyramid(im, sigma=1, prNum=6):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]
    
    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma)
    
    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width/2), int(height/2)))
        pyramids.append(G)
    
    
    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i-1:
                im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im

    return pyramids

def foveat_img(im, fixs):
    """
    im: input image
    fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    
    This function outputs the foveated image with given input image and fixations.
    """
    sigma=0.248
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape
    
    # compute coef
    p = 1 # blur strength
    k = 3 # size of foveation
    alpha = 5 # also size?

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)
    
    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(theta))

    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma

    omega[omega>1] = 1

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))

    # M
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))

    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i-1][ind]

        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]

    print('num of full-res pixel', np.sum(Ms[0] == 1))
    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])

    im_fov = im_fov.astype(np.uint8)
    return im_fov

"""
Original __main__
"""
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Wrong format: python retina_transform.py [image_path]")
#         exit(-1)

#     im_path = sys.argv[1]
#     im = cv2.imread(im_path)
#     print(im)
#     # im = cv2.resize(im, (512, 320), cv2.INTER_CUBIC)
#     xc, yc = int(im.shape[1]/2), int(im.shape[0]/2)

#     im = foveat_img(im, [(xc, yc)])

#     cv2.imwrite(im_path.split('.')[0]+'_RT.jpg', im)

"""
Johan's __main__
""" 

# # TODO given I use resizing and cropping no need to repeat this function for every image
# def fill_fov(im):
#     fov_points = []
#     indices = []

#     # loop over the image row by row
#     for i in range(1, 10, 2):
#         for j in range(1, 10, 2):
#             indices.append((i,j))
#             fov_points.append((int(im.shape[1]*(i/10)), int(im.shape[0]*(j/10))))

#     # also return indices because those are needed to correctly name the output files
#     return indices, fov_points

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Wrong format: python retina_transform.py [image_path]")
#         exit(-1)

#     folder_path = sys.argv[1]
#     im_paths = os.listdir(folder_path)      

#     # image transformation - parameters
#     # this is needed because otherwise foveation point outside of cropped area
#     input_size = 224
#     resize_size = int(input_size/0.875) #256 for input_size 224
#     margin = int((resize_size - input_size)/2)

#     for im_path in im_paths:
#         start = time.time()
#         im = cv2.imread(folder_path + '/' + im_path)

#         # image transformation - application
#         resized_im = cv2.resize(im, (256, 256))
#         cropped_im = resized_im[margin:-margin, margin:-margin]

#         dirname = folder_path + '/' + im_path.split('.')[0]
#         os.mkdir(dirname)

        

#         indices, fov_points = fill_fov(cropped_im)

#         for i, fov_point in enumerate(fov_points):
#             xc, yc = fov_point
#             temp = foveat_img(cropped_im, [(xc, yc)])
#             #adding a red dot so that spotting the foveation point is easier
#             #cv2.circle(temp, (xc, yc), 5, (0, 0, 255) , -1)

#             # TODO make sure the filenames are more robust
#             # for the specific application of a pytorch dataloader, make sure the names are getting ordered correctly.
#             cv2.imwrite(dirname + '/' + im_path.split('.')[0] + '_' + str(indices[i]) +'_RT.jpg', temp)

#         end = time.time()
#         print("done with: " + im_path + " in " + str(end-start) + " seconds.")

"""
Second main
"""
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong format: python retina_transform.py [image_path]")
        exit(-1)

    folder_path = sys.argv[1]
    im_paths = os.listdir(folder_path)      

    # image transformation - parameters
    # this is needed because otherwise foveation point outside of cropped area
    input_size = 224
    resize_size = int(input_size/0.875) #256 for input_size 224
    margin = int((resize_size - input_size)/2)

    # dirname = 'E:\\ILSVRC2017\\newimages\\notfoveated\\n01531178'
    dirname = 'E:\ILSVRC2017\multiplefoveationsTEST\\result'
    os.mkdir(dirname)

    for im_path in im_paths:
        start = time.time()
        im = cv2.imread(folder_path + '/' + im_path)

        # image transformation - application
        resized_im = cv2.resize(im, (256, 256))
        cropped_im = resized_im[margin:-margin, margin:-margin]

        # xc, yc = int(cropped_im.shape[1]/2), int(cropped_im.shape[0]/2)
        # foveated_im = foveat_img(cropped_im, [(xc, yc)])
        # cv2.imwrite(dirname + '/' + im_path.split('.')[0] + '.jpg', cropped_im)

        xc1, yc1 = (int(cropped_im.shape[1]*(5/10)), int(cropped_im.shape[0]*(5/10)))
        xc2, yc2 = (int(cropped_im.shape[1]*(7/10)), int(cropped_im.shape[0]*(5/10)))

        # foveated_im = foveat_img(cropped_im, [(xc1, yc1)])
        # cv2.circle(foveated_im, (xc1, yc1), 5, (0, 0, 255) , -1)
        # cv2.imwrite(dirname + '/' + im_path.split('.')[0] + '_1.jpg', foveated_im)

        # foveated_im = foveat_img(cropped_im, [(xc1, yc1), (xc2, yc2)])
        # cv2.circle(foveated_im, (xc1, yc1), 5, (0, 0, 255) , -1)
        # cv2.circle(foveated_im, (xc2, yc2), 5, (0, 0, 255) , -1)
        # cv2.imwrite(dirname + '/' + im_path.split('.')[0] + '_2.jpg', foveated_im)
        
        # cv2.imwrite(dirname + '/' + im_path.split('.')[0] + '_RT.jpg', foveated_im)


        end = time.time()
        print("done with: " + im_path + " in " + str(end-start) + " seconds.")