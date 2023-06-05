# python-watershed
image processing, labeling a binary mask, etc...

This routine implements the FogBank algorithm from J.Chalfoun et al., "Fog Bank: A Single Cell Segmentation across Multiple Cell Lines and Image Modalities", BMC BioInformatics, 2014 (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-014-0431-x).

The routine takes a binary segmented mask and separates out touching objects (see image below). The FogBank routine avoids oversegmentation from typical watershed algorithms.

![watershed](https://github.com/zbenson94/python-watershed/assets/55454108/c8bc3d7c-2df6-4bf3-a018-b681748dc70d)



# Usage:

from python_watershed import FogBank

import skimage.io as skio

######  # Common input parameters
fogbank_parms = {
  'min_size': 8,
  'min_object_size': 50,
  'erode_size': 2
}

###### # Load binary image
img = skio.imread(path_to_image)


fb  = FogBank(img, **fogbank_parms)



img_labeled = fb.run()



