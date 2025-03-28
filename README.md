## Download Dataset 

Sketch-Voxel ShapeNet dataset can be downloaded form the following link:

```
https://drive.google.com/file/d/1aXug8PcLnWaDZiWZrcmhvVNFC4n_eAih/view?pli=1
```

## Get Started

To train SkFC-AE, you can simply use the following command:

```
python runner.py --category=<category>
```

To test SkFC-AE, you can use the following command:

```
python runner.py --test --weights=/path/to/pretrained/model.pth
```

## Note

All the results (all comparison models and ablation experiments) in our paper are on the VAL set, and the TEST set is only used to generate visual 3D shapes.


## License

This project is open sourced under MIT license.

