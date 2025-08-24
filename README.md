# Gömböc Infinite Search (Multi-Threaded)

This project implements an **infinite search for Gömböc-like shapes** using **multi-threading**, evaluating candidate shapes, and generating high-quality STL files for 3D printing.

A Gömböc is a convex, homogeneous object with **exactly one stable and one unstable equilibrium point**. This script searches for such shapes using spherical harmonics as a parametric surface representation.

---

## Features

- **Multi-threaded infinite search** for Gömböc candidates.
- Evaluates convexity and equilibrium points of shapes.
- Saves high-quality STL files for 3D printing.
- Prints bounding box dimensions and candidate information.
- Adjustable sample density for surface points and gravity directions.

---

## Requirements

- Python 3.10+
- Packages:
  - `numpy`
  - `scipy`
  - `numpy-stl`

Install dependencies with:

```bash
pip install numpy scipy numpy-stl
```
## Usage

Run the script from the command line:

```
python gomboc_search_high_quality_mt.py
```

## Options

num_threads: Adjust the number of threads to match your CPU cores:
infinite_search_mt(num_threads=8)

```
infinite_search_mt(num_threads=8)

```

Output directory: STL files are saved in the candidates/ folder. Filenames follow the pattern:

```
candidates/gomboc_1.stl
candidates/gomboc_2.stl
```

Bounding box: For each found candidate, the script prints:
```
Bounding box: x=..., y=..., z=...
```

This helps to estimate the scale before 3D printing.

## How It Works

- Surface Parameterization: Shapes are represented by a sum of spherical harmonics.
- Support Function Evaluation: The script converts harmonic coefficients to support function values over the unit sphere.
- Surface Points Generation: Converts support function into 3D surface points using local tangent approximations.
- Convexity Check: Verifies that the candidate shape is strictly convex.
- Equilibrium Classification: Computes stable (S), saddle (H), and unstable (U) equilibrium points for multiple gravity directions.
- Candidate Selection: Only convex shapes with 1 stable, 0 saddle, 1 unstable equilibrium are considered valid Gömböc candidates.
- STL Generation: Delaunay triangulation of the surface points generates a high-quality STL file suitable for 3D printing.

## Notes

- This script runs infinitely until manually stopped.
- Increasing sample_normals and gravity_samples improves accuracy but increases computation time.
-Multi-threading significantly accelerates the search on multi-core CPUs.

## Example Output

```
Candidate #1: vec[2]=0.012345, vec[6]=-0.045678
Bounding box: x=0.512, y=0.498, z=0.489
Saved STL: candidates/gomboc_1.stl
```
