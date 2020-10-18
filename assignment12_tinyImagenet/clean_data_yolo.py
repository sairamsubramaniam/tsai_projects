
import copy
import csv
import json
import os
import numpy as np
import pandas as pd
from PIL import Image


def get_image_dims(imgs_folder_path, img_details_csv_path):
    filenames = os.listdir(imgs_folder_path)
    csv_contents = [["filename","img_width","img_height"]]
    for fl in filenames:
        img = Image.open(os.path.join(imgs_folder_path, fl))
        w, h = img.size
        csv_contents.append([fl, w, h])

    with open(img_details_csv_path, "w") as otfl:
        wt = csv.writer(otfl)
        wt.writerows(csv_contents)


def get_bb_details(annot_json_path, bb_details_csv_path):
    bigjson = json.load(open(annot_json_path))

    rows = []
    for fkey, fval in bigjson.items():
        row = {}
        row["filename"] = fval["filename"]
        for reg in fval["regions"]:
            r = copy.deepcopy(row)
            r["bb_cent_x"] = reg["shape_attributes"]["x"]
            r["bb_cent_y"] = reg["shape_attributes"]["y"]
            r["bb_width"] = reg["shape_attributes"]["width"]
            r["bb_height"] = reg["shape_attributes"]["height"]
            r["labels"] = reg["region_attributes"].get("labels", "Not Labelled")
            rows.append(r)

    with open(bb_details_csv_path, "w") as otfl:
        wt = csv.DictWriter(otfl, fieldnames=["filename", "labels", 
                                              "bb_cent_x", "bb_cent_y",
                                              "bb_width", "bb_height"]
                           )
        wt.writeheader()
        wt.writerows(rows)



def merge_img_bb_details(img_details_csv, bb_details_csv, final_csv, blocks=13):
    df_img = pd.read_csv(img_details_csv)
    df_bb = pd.read_csv(bb_details_csv)
    df = pd.merge(df_img, df_bb, how="inner", on="filename")

    # Normalize BB Width & Height
    df["norm_bb_width"] = df["bb_width"] / df["img_width"]
    df["norm_bb_height"] = df["bb_height"] / df["img_height"]

    #df["bb_cent_block_x"] = df["

    df.to_csv(final_csv)


if __name__ == "__main__":
    imgs_folder_path = ("/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/"
                       "assignment12_tinyImagenet/images_for_yolo")
    img_prop_csv_path = ("/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/"
                         "assignment12_tinyImagenet/image_details.csv")

    annot_json_path = ("/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/"
                       "assignment12_tinyImagenet/s12_annotations_json.json")
    bb_prop_csv_path = ("/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/"
                       "assignment12_tinyImagenet/bb_details.csv")


    final_csv = ("/home/sai/Documents/repos_and_projects/personal_projects/tsai_projects/"
                       "assignment12_tinyImagenet/clustering_input.csv")


    get_image_dims(imgs_folder_path=imgs_folder_path, img_details_csv_path=img_prop_csv_path)
    get_bb_details(annot_json_path=annot_json_path, bb_details_csv_path=bb_prop_csv_path)
    merge_img_bb_details(img_details_csv=img_prop_csv_path, 
                         bb_details_csv=bb_prop_csv_path, 
                         final_csv=final_csv)

    # Cleanup:
    os.remove(img_prop_csv_path)
    os.remove(bb_prop_csv_path)
