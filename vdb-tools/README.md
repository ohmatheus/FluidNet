# vdb-tools

Pipeline for generating training data from Blender fluid simulations. Takes 3D volumetric VDB files and converts them into 2D NPZ datasets ready for PyTorch.

## Setup

You'll need a conda environment because the VDB-related libraries are only available through conda, not pip.

Create the environment from the provided file:

```bash
conda env create -f environment.yml
conda activate vdb-env
```

Configure your Blender installation path:

```bash
cp .env.example .env
# Edit .env and set BLENDER_PATH to your Blender 4.5 LTS installation
```

## Workflow

### 1. Generate Blender Simulations

Creates randomized fluid scenarios using headless Blender:

```bash
python create_simulations.py --count 80 --resolution 128 --min-frames 110 --max-frames 130 --workers 2
```

This generates VDB cache files (volumetric data) and Alembic files (mesh animations) in the data directory.

### 2. Convert to NPZ Format

Projects 3D simulations onto 2D planes and packages them as training-ready NPZ files:

```bash
python vdb_to_numpy.py --resolution 128 --workers 6
```

Outputs compressed NPZ sequences with density, velocity, and mask fields.

### 3. Debug/Visualize (Optional)

Render NPZ fields as PNG images for inspection:

```bash
python render_npz_field_to_png.py --field density
```

If you have ffmpeg installed, you can preview the PNG sequence:

```bash
ffplay -framerate 15 -i 'frame_%04d.png'
```

## What's Happening Under the Hood

- **create_simulations.py**: Calls Blender in headless mode to bake Mantaflow fluid simulations with randomized emitters and colliders
- **vdb_to_numpy.py**: Reads VDB grids, averages across the Y-axis to project 3D→2D, extracts mesh masks from Alembic files, and saves everything as NPZ
- **render_npz_field_to_png.py**: Useful for checking if your data looks reasonable before training

The whole pipeline is designed to be run in batch - generate a bunch of simulations, convert them all, then feed them to the ML training pipeline.
