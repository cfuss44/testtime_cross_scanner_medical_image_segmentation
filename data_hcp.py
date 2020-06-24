import numpy as np
import data_brain_hcp

# ============================
# main function that redirects the data loading to the requested anatomy and dataset
# ============================   
def load_data(input_folder,
                preprocessing_folder,
                protocol,
                idx_start,
                idx_end):

    # protocol: T1 or T2
      
    # =====================================
    # Set the desired resolution and image size here.
    # Currently, the pre-processing is set up for rescaling coronal slices to this target resolution and target image size,
    # while keeping the resolution perpendicular to the coronal slices at its original value.
    # =====================================
    image_size = (256, 256)
    target_resolution = (0.7, 0.7)
    
    # =====================================
    # The image_depth parameter indicates the number of coronal slices in each image volume.
    # All image volumes are cropped from the edges or padded with zeros, in order to ensure that they have the same number of coronal slices,
    # keeping the resolution in this direction the same as in the original images.
    # =====================================
    image_depth = 256
        
    # =====================================
    # pre-processing function that returns a hdf5 handle containing the images and labels
    # =====================================
    data_brain = data_brain_hcp.load_and_maybe_process_data(input_folder = input_folder,
                                                            preprocessing_folder = preprocessing_folder,
                                                            idx_start = idx_start,
                                                            idx_end = idx_end,             
                                                            protocol = protocol,
                                                            size = image_size,
                                                            depth = image_depth,
                                                            target_resolution = target_resolution)
    
    # =====================================  
    # Extract images and labels from the hdf5 handle and return these.
    # If desired, the original resolutions of the images and original image sizes, as well as patient ids are also stored in the hdf5 file.
    # =====================================  
    images = data_brain['images']
    labels = data_brain['labels']

    return np.array(images), np.array(labels)