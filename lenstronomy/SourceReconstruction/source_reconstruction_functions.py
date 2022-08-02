import numpy as np

import lenstronomy.Util.util as util

__all__ = ['SourceReconstruction']

class SourceReconstruction(object):
    """
    This class defines useful functions for the purpose of pixellated source plane reconstrcution.
    
    """
    
    def __init__(self, data_class, lens_model_class, kwargs_data, kwargs_lens):
        
        # get the width of the image (numPix) and the length of one pixel (deltaPix) from kwargs_data
        self.numPix = len(kwargs_data['image_data'])
        self.deltaPix = kwargs_data['transform_pix2angle'][0,0]
        self.deltaPixsq = self.deltaPix**2
        
        # the (x,y) coordinate at the (0,0) pixel
        self.minx = kwargs_data['ra_at_xy_0']
        self.miny = kwargs_data['dec_at_xy_0']
        
        # give the coordinate griddings for source and image planes
        # x_grid, y_grid is the source plane (x,y) coordinate gridding
        # beta_x_grid, beta_y_grid is also the source plane (x,y) coordinate, but lensed to the image plane
        
        x_grid_temp, y_grid_temp = data_class.pixel_coordinates
        self.x_grid = util.image2array(x_grid_temp)
        self.y_grid = util.image2array(y_grid_temp)
        self.beta_x_grid, self.beta_y_grid = lens_model_class.ray_shooting(self.x_grid, self.y_grid, kwargs=kwargs_lens)
        
        try:
            kwargs_data['primary_beam']
        except:
            kwargs_data['primary_beam'] = None
        self.primary_beam = kwargs_data['primary_beam']


    def give_source_plane_gridding(self):
        # give x,y coordinates for each pixal of the source plane
        # output is x coordinate array and y coordinate array
        return self.x_grid, self.y_grid
    
    def give_image_plane_gridding(self):
        # give the source plane x,y coordinates mapped to the lens plane pixel
        # output is beta_x coordinate array and beta_y coordinate array
        return self.beta_x_grid, self.beta_y_grid

    # we would like to make each pixel at the source plane and its lensed counterpart as sparse matrces to save computational time and storage
    # one sparse matrix is in the representation [[row1,col1,value1],[row2,col2,value2],...] for non-zero elements
    # for eg, a sparse matrix, [[10,20,1]] means that this is an array with the (10,20) entry being 1 while all other entries being zero.
    # below are some functions for sparse matrix
    
    def sparse_to_array(self,sparse):
        # convert a sparse array to a 2d array
        # input is one sparse matrix, output the reverted array
        image = np.zeros((self.numPix,self.numPix))
        num_of_elements = len(sparse)
        for i in range(num_of_elements):
            image[sparse[i][0],sparse[i][1]] = sparse[i][2]
        return image

    def product_sp_ordinary(self,sparse,ordinary):
        # element wise product of a sparse matrix with an ordinary matrix
        sum_temp = 0
        num_element = len(sparse)
        for i in range(num_element):
            sum_temp += sparse[i][2] * ordinary[sparse[i][0],sparse[i][1]]
        return sum_temp
    
    def sparse_convolution_product(self,sp1,sp2,kernel):
        # convolution product of two sparse matrices
        # the kernel should be an odd times odd size square array
        # make sure the kernel has its centre at the central pixel
        inner_product = 0
        num_elements_1 = len(sp1)
        num_elements_2 = len(sp2)
        kernel_center = int(len(kernel)/2)
        for i in range(num_elements_1):
            for j in range(num_elements_2):
                inner_product += sp1[i][2] * sp2[j][2] * kernel[kernel_center + (sp1[i][0]-sp2[j][0]),kernel_center + (sp1[i][1]-sp2[j][1])]
        return inner_product

    def sparse_convolution(self,sp,kernel):
        # convolution of a sparse matrix
        # the kernel should be an odd times odd size square array
        # make sure the kernel has its centre at the central pixel
        # the output is the convolved matrix, in a non-sparse form. The size of the output image is the size given by numPix
        kernel_center = int(len(kernel)/2)
        convolved = np.zeros((self.numPix,self.numPix))
        num_element_sparse = len(sp)
        for i in range(num_element_sparse):
            convolved += sp[i][2] * kernel[kernel_center - sp[i][0]:kernel_center - sp[i][0]+self.numPix:1,
                                           kernel_center - sp[i][1]:kernel_center - sp[i][1]+self.numPix:1]
        return convolved
    
    
    def lens_a_source_plane_pixel(self,row,col):
        # lensing a source plane pixel to the image plane
        # input is the row and col of the pixel at the source plane
        # the output is in a sparse matrix form
        
        ps_x = self.minx + self.deltaPix * col
        ps_y = self.miny + self.deltaPix * row
        
        x_lower = ps_x - self.deltaPix
        x_upper = ps_x + self.deltaPix
        y_lower = ps_y - self.deltaPix
        y_upper = ps_y + self.deltaPix
        
        lensed_image_sparse = []
        
        for i in range(self.numPix):
            for j in range(self.numPix):
                
                n_beta = i*self.numPix + j
    
                cor_beta_x = self.beta_x_grid[n_beta]
                
                if cor_beta_x < x_lower or cor_beta_x > x_upper:
                    continue
                else:
                    cor_beta_y = self.beta_y_grid[n_beta]
                    if cor_beta_y < y_lower or cor_beta_y > y_upper:
                        continue
                    else:
                        value = (self.deltaPix - np.abs(ps_x - cor_beta_x))* (self.deltaPix - np.abs(ps_y - cor_beta_y))/(self.deltaPixsq)
                        if self.primary_beam is not None:
                            value *= self.primary_beam[i,j]
                        lensed_image_sparse.append([i,j,value])
                        
        return lensed_image_sparse
    


    # below are two functions mapping an image of an array form (length x length array) from the source plane to the (lensed)image plane 
    # and from the image plane to the source plane
    
    def lens_an_image_by_rayshooting(self,image):
        # lensing a pixelated image from source plane to image plane
        # the input is the source plane image
        # the output is the image plane image
        # this method is approximate. the flux between well defined pixels are gotten by interpolating
        
        nx,ny = np.shape(image)
        if nx != self.numPix or ny != self.numPix:
            raise ValueError('The image size should be the same as the defined one of the data class!')
            
        lensed_image = np.zeros((nx,ny))
    
        for i in range(nx):
            for j in range(ny):
                n_beta = i*ny + j

                cor_beta_x = self.beta_x_grid[n_beta]
                cor_beta_y = self.beta_y_grid[n_beta]

                n_x = int((cor_beta_x-self.minx)/self.deltaPix)
                n_y = int((cor_beta_y-self.miny)/self.deltaPix) 
            
                if n_x >= nx-1 or n_y >= ny-1 or n_x<0 or n_y <0:
                    lensed_image[i,j] = 0
                else:
                    weight_upper_left = np.abs(self.miny + n_y*self.deltaPix + self.deltaPix - cor_beta_y)*np.abs(self.minx + n_x*self.deltaPix + self.deltaPix - cor_beta_x)/(self.deltaPix**2)
                    weight_upper_right = np.abs(self.miny + n_y*self.deltaPix + self.deltaPix - cor_beta_y)*np.abs(self.minx + n_x*self.deltaPix - cor_beta_x)/(self.deltaPix**2)
                    weight_lower_left = np.abs(self.minx + n_x*self.deltaPix + self.deltaPix - cor_beta_x)*np.abs(self.miny + n_y*self.deltaPix - cor_beta_y)/(self.deltaPix**2)
                    weight_lower_right = np.abs(self.minx + n_x*self.deltaPix - cor_beta_x)*np.abs(self.miny + n_y*self.deltaPix - cor_beta_y)/(self.deltaPix**2)
                        
                    lensed_image[i,j] = image[n_y,n_x] * weight_upper_left + image[n_y,n_x+1] * weight_upper_right + (
                           image[n_y+1,n_x] * weight_lower_left + image[n_y+1,n_x+1] * weight_lower_right)
                
        return lensed_image
    
    
    def delens_an_image_by_rayshooting(self,lensed_image):
        # delensing a pixelated image from the image plane to the source plane
        # the input is the image plane image
        # the output is the source plane image
        # this method is approximate.
        # this is NOT the perfect inverse function of the function lens_an_image_by_rayshooting(). They use different approximation method.
        
        nx,ny = np.shape(lensed_image)
        if nx != self.numPix or ny != self.numPix:
            raise ValueError('The image size should be the same as the defined one of the data class!')
        
        n_pixel = nx*ny
        
        source_image_weighted_flux = [[[0]*2 for _ in range(1)] for __ in range(n_pixel)]
        source_image = np.zeros((nx*ny))
        
        for i in range(ny):
            for j in range(nx):
                
                n_beta = i*nx + j
                
                cor_x_im = self.beta_x_grid[n_beta]
                cor_y_im = self.beta_y_grid[n_beta]
                
                n_x_src_non_int = (cor_x_im - self.minx)/self.deltaPix
                n_y_src_non_int = (cor_y_im - self.miny)/self.deltaPix
                
                n_x_src = int(n_x_src_non_int)
                n_y_src = int(n_y_src_non_int)
                
                if n_x_src<0 or n_y_src<0 or n_x_src>=nx-1 or n_y_src>= ny-1:
                    continue
                
                w00 = (1 + n_x_src - n_x_src_non_int)*(1 + n_y_src - n_y_src_non_int)
                w01 = (1 + n_x_src - n_x_src_non_int)*(n_y_src_non_int - n_y_src)
                w10 = (n_x_src_non_int - n_x_src)*(1 + n_y_src - n_y_src_non_int)
                w11 = (n_x_src_non_int - n_x_src)*(n_y_src_non_int - n_y_src)
                
                source_image_weighted_flux[n_y_src*nx + n_x_src].append([w00,lensed_image[i,j]])
                source_image_weighted_flux[n_y_src*nx + n_x_src + 1].append([w10,lensed_image[i,j]])
                source_image_weighted_flux[n_y_src*nx + nx + n_x_src].append([w01,lensed_image[i,j]])
                source_image_weighted_flux[n_y_src*nx + nx + n_x_src + 1].append([w11,lensed_image[i,j]])
                
        for i in range(n_pixel):
            count_image_to_source = len(source_image_weighted_flux[i])
            if count_image_to_source == 1:
                source_image[i] = 0
            else:
                temp_weight = 0
                temp_flux = 0
                for j in range(count_image_to_source-1):
                    temp_weight += source_image_weighted_flux[i][j+1][0]
                    temp_flux += source_image_weighted_flux[i][j+1][0] * source_image_weighted_flux[i][j+1][1]
                source_image[i] = temp_flux/temp_weight
        
        source_image = util.array2image(source_image)
                    
        return source_image