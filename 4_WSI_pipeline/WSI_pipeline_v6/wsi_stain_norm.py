#FUNCTIONS FOR STAIN NORMALIZATION

import staintools

#Initialize BrightnessStandardizer
standardizer = staintools.BrightnessStandardizer()

#Initialize StainNormalizer "macenko"
stain_norm = staintools.StainNormalizer(method='macenko')