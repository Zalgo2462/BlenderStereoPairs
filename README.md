# Blender Stereo Pairs
#### Generates a stereo pair dataset using Blender with images, calibration details, and depth maps.

---

This project generates a stereo pair dataset using Blender with images, calibration details, and depth maps.
The generated camera positions are evenly sampled across a spherical segment, with each camera pointing
towards the center of the sphere. Each pair of cameras are considered a stereo pair if the central angle
between the two cameras as measured from the center of the sphere is within a given threshold.

#### Dataset Generation
Run with `blender <blend file> -P blender_pairs_render.py -- [args]`.
Unless `--output_dir <directory>` is passed as part of `args`, the results will be written to `./blender_pairs`.

The following packages must be available in the Blender Python environment. See https://blender.stackexchange.com/a/122337 
for instructions on installing Python packages in Blender.
- `numpy`
- `scipy`

The list of stereo pairs can be found in `blender_pairs/pairs.csv`. Each row represents a stereo pair
with the format `<image 1 camera name>, <image 1 frame number>, <image 2 camera name>, <image 2 frame number>`.
Each stereo pair addresses two images, depth maps, intrinsic calibration matrices, and extrinsic calibration matrices.

The image data and depth maps for each (camera, frame) tuple can be found in EXR format at
 `blender_pairs/EXR/<camera name>/<frame number>.exr`. The RGB-D data is stored as 32 bit floats in the channels
 `Image.R`, `Image.G`, `Image.B`, and `Depth.V`. `10000000000` is used in `Depth.V` when there is no valid depth data.

The 3x3 intrinsic calibration matrix for each camera can be found in numpy txt format at
 `blender_pairs/K/<camera name>.txt`.

The 3x4 extrrinsic calibration matrix for each (camera, frame) tuple can be found in numpy txt format at
 `blender_pairs/RT/<camera name>/<frame number>.txt`.

The arguments passed to the script are stored in `blender_paris/params.json`.

#### Reading the Dataset
To read the resulting data, import `blender_pairs_render.py`.

Read a `pairs.csv` file by calling `read_pairs_index(fd: IO) -> List[PairIndex]`.

Load the data associated with each half of a stereo pair by calling
`load(self, data_root: str) -> Tuple[BlenderImage, BlenderImage]`.

Each `BlenderImage` contains all of the information pertaining to half of a stereo pair.
```python
# BlenderImage represents a loaded half of a stereo pair
BlenderImage = collections.namedtuple("BlenderImage", [
    "image",  # numpy uint8 bgr image
    "depth_map",  # numpy float32 depth map. 10000000000 marks no data
    "intrinsic_matrix",  # 3x3 numpy float32 intrinsic calibration matrix
    "extrinsic_matrix",  # 3x4 numpy float32 extrinsic calibration matrix
    "name"  # "<camera_name>:<frame_number>"
])
```

Example of plotting images with matplotlib:
```python
import matplotlib.pyplot as plt
from blender_pairs_read import read_pairs_index

with open("./blender_pairs/pairs.csv", 'r') as fd:
    pairs_index = read_pairs_index(fd)
    img1, img2 = pairs_index[0].load("./blender_pairs")
    img1_rgb = img1.image[:, :, ::-1]  # convert bgr channel format to rgb
    plt.imshow(img1_rgb)
```