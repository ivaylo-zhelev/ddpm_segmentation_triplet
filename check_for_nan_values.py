from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np

images_path = Path("/mnt/c/Users/ivayl/Downloads/289RM7EZ02HD117WJ5IM/Starvation_1_48H_-_For_ASMAA-Stitched/Stacks_crops_complete_window100_w320_h320")
segm_path = Path("/mnt/c/Users/ivayl/Downloads/289RM7EZ02HD117WJ5IM/Starvation_1_48H_-_For_ASMAA-Stitched/Masks_crops_complete_window100_w320_h320")

def main():
    for image in tqdm(list(images_path.rglob("*.png")) + list(segm_path.rglob("*.png"))):
        im = Image.open(image)
        im = np.array(im) / 255

        if np.any(np.isnan(im)):
            print(f"Faulty image at {image.name}")

if __name__ == "__main__":
    main()