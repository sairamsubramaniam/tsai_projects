
"""
Noting down steps followed:

1. The ultralytics weights file - made a copy in drive and moved it to the required folder in drive itself
2. Locally create the folder structure as mentioned here https://github.com/theschoolofai/YoloV3
3. Created custom.data and custom.names locally inside customdata folder
4. Changed Images/ to "data/customdata/images" in train.txt
5. Changed .txt to .jpg in train.txt
6. In train.txt, file named "Aimgg_005.jpg" is a typo - so rename it to Aimg_005.jpg
7. In test.txt, file names ImageYolo.jpg does not exists, so remove it
8. In test.txt, file names M/img_010 does not exists - so rename it to Mimg_010

"""

import os
import shutil
import zipfile as zp

from PIL import Image




ZOHEB_ZIPFILE_PATH = "/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/assignment13_yolo/YoloV3_Dataset.zip"
CUSTOMDATA_FOLDERPATH = "/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/assignment13_yolo/images_structured/customdata"

ARRANGED_IMGFOLDER = CUSTOMDATA_FOLDERPATH + "/images"
ARRANGED_LBLFOLDER = CUSTOMDATA_FOLDERPATH + "/labels"


# Code for checking if any value in label texts is above 1.000




#  Code for checking if file exists:


with open("./data/customdata/train.txt","r") as infl:
    for k in infl:
        k = k.strip()
        if not os.path.exists( "/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/assignment13_yolo/" + k.replace("./","") ):
            print( "/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/assignment13_yolo/" + k.replace("./","") )




def make_filepath_list(imgfolder, lblfolder):
    image_files = [ os.path.splitext(l)[0] for l in os.listdir(imgfolder) ]
    label_files = [ os.path.splitext(l)[0] for l in os.listdir(lblfolder) ]

    image_names = [ os.path.splitext(l)[0] for l in os.listdir(imgfolder) ]

    



def rearrange_images_labels_and_zip(zipfile_path, customdata_folderpath):

    extract_path, extract_folder = os.path.split(zipfile_path)
    extract_folder = extract_folder.replace(".zip","")

    # Extract contents from zip
    with zp.ZipFile(zipfile_path, "r") as zip:
        zip.extractall()

    # Define relevant file paths
    imgfolder = extract_folder + "/Images"
    lblfolder = extract_folder + "/Labels"
    train_file_list = extract_folder + "/train.txt"
    test_file_list = extract_folder + "/test.txt"
    name_list = extract_folder + "/classes.txt"

    dest_imgfolder = ARRANGED_IMGFOLDER
    dest_lblfolder = customdata_folderpath + "/labels"
    dest_train_file_list = customdata_folderpath + "/train.txt"
    dest_test_file_list = customdata_folderpath + "/test.txt"
    dest_name_list = customdata_folderpath + "/custom.names"
    

    shutil.copytree(imgfolder, dest_imgfolder, dirs_exist_ok=True)
    shutil.copytree(lblfolder, dest_lblfolder, dirs_exist_ok=True)
    shutil.copy(train_file_list, dest_train_file_list)
    shutil.copy(test_file_list, dest_test_file_list)
    shutil.copy(name_list, dest_name_list)



def rename_image_exts(arranged_imgfolder):

    image_filenames = os.listdir(arranged_imgfolder)

    for fl in image_filenames:
        fil, ex = os.path.splitext(fl)
        if ex != ".jpg":
            print(fl)

            if ex == ".png":
                png_to_jpg(image_filepath=os.path.join(arranged_imgfolder, fl))
            else:
                os.rename( os.path.join(arranged_imgfolder, fl), 
                           os.path.join(arranged_imgfolder, fil+".jpg") )



def png_to_jpg(image_filepath):
    im = Image.open(image_filepath)
    rgb_im = im.convert("RGB")
    rgb_im.save(image_filepath.replace(".png",".jpg"))
    os.remove(image_filepath)


def clip_to_ones_in_labels():
    pass



if __name__ == "__main__":
    rearrange_images_labels_and_zip(zipfile_path=ZOHEB_ZIPFILE_PATH, 
                                    customdata_folderpath=CUSTOMDATA_FOLDERPATH)
#     rename_image_exts(arranged_imgfolder=ARRANGED_IMGFOLDER)




