# python-watershed
image processing, labeling a binary mask, etc...

This routine implements the FogBank algorithm from J.Chalfoun et al., "Fog Bank: A Single Cell Segmentation across Multiple Cell Lines and Image Modalities", BMC BioInformatics, 2014 (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-014-0431-x).

The routine takes a binary segmented mask and separates out touching objects (see image below). The FogBank routine avoids oversegmentation from typical watershed algorithms.

![watershed](https://github.com/zbenson94/python-watershed/blob/ed9a8231d32823891dea603d2c4a609eb2b37edc/watershed.png)


# Usage:
```python
from python_watershed import FogBank

import skimage.io as skio

fogbank_parms = {
  'min_size':         8,
  'min_object_size':  50,
  'erode_size':       2
}
img          = skio.imread(path_to_image)
img_labeled  = FogBank(img, **fogbank_parms).run()
```

# Installation Instructions (if python is installed)
```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```



