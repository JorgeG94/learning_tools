First, get a meson install 

```
git clone git@github.com:mesonbuild/meson.git
cd meson 
./packaging/create_zipapp.py --outfile meson.pyz --interpreter '$YOUR_PATH_TO_WHERE_PYTHON_IS/bin/python' .
cp meson.pyz ~/.local/bin
```

module load nvidia-hpc-sdk/25.1
pip install --no-build-isolation -e .
nsys profile --stats=true python a.py
