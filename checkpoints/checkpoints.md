# Checkpoints

The required model checkpoints are hosted on Google Drive and are downloaded automatically at runtime using `gdown`.

Required files:
- classifier.pth
- localizer.pth
- unet.pth

These files are used to initialize the unified multi-task model.

Notes:
- The repository does not store the .pth files directly.
- Filenames must remain exactly:
  - classifier.pth
  - localizer.pth
  - unet.pth
- models/multitask.py handles automatic download.
