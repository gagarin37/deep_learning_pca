
#Count number of patches with tumor and benign tissue
#to further calculate tumor volume / square
count_be = 0
count_tu = 0
for he in range(wsi_map_bin.shape[0]):
    for wi in range(wsi_map_bin.shape[1]):
        if wsi_map_bin [he,wi] == 0:
            print ("background")
        elif wsi_map_bin [he,wi] == 1111:
            print ("benign")
            count_be = count_be + 1
        else:
            print ("tumor")
            count_tu = count_tu + 1
   

