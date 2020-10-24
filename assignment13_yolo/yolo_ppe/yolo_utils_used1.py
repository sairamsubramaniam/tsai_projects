
import os

fldr = "./data/customdata/labels"
labels = os.listdir(fldr)

for fl in labels:
    with open( os.path.join(fldr, fl) , "r") as infl:
        for k in infl:
            k_l = k.split()[1:]
            for num in k_l:
                if float(num) > 1.0:
                    print(fl, " | ", k, " | ", num)







with open("./data/customdata/train.txt","r") as infl:
    for k in infl:
        k = k.strip()
        if not os.path.exists( "/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/assignment13_yolo/" + k.replace("./","") ):
            print( "/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/assignment13_yolo/" + k.replace("./","") )



import csv
from PIL import Image


lols = []
with open("./data/customdata/test.txt","r") as infl:
    for k in infl:
        k = k.strip()
        img = Image.open( "/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/assignment13_yolo/" + k.replace("./","") )
        lols.append( list(img.size) )


with open("test.shapes", "w") as otfl:
    wt = csv.writer(otfl)
    wt.writerows(lols)





