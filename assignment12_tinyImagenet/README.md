
# Modelling Tiny ImageNet And Image Annotations

### The requirements in detail:
  

The VIA tool outputs annotations in a json and this is the format of the json file:

  "image_filename+filesize": {

    "filename": "Actual Filename of the image1_valid4.jpeg",

    "size": Size (in bytes) of the image file,

    "regions - (List of bounding boxes' annotations on the image)": [
      {
        "shape_attributes - (Attributes of this bounding box)": {
          "name - (Tell what shape is the bounding box)": "rect",
          "x - (X coordinate of the centre of this bounding box)": 121,
          "y - (Y coordinate of the centre of this bounding box)": 55,
          "width - (Width of the bounding box)": 28,
          "height - (Height of the bounding box)": 14
        },
        "region_attributes": {
          "labels - (Name of the attribute e.g "Label", "Class" etc.)": "hardhat - (Value of the attribute - the actual class of this object covered by the bounding box)"
        }
      },
      { Next bounding box},
      { Next bounding box in the image },
      .....
      { ... }
    ],

    "file_attributes - (More info about the image file)": {
      "caption - (Caption for the image)": "",
      "public_domain - (If this image has been made available on the internet)": "no",
      "image_url - (Url of the image)": ""
    }
  },

  "NEXT IMAGE": {}
  "NEXT IMAGE": {}
  ......
}
       


