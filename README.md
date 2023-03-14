# DOCS
FIXING ISSUES
### `train_images`

array of train images 28x28

### `train_labels`

array of labels for each image form train_images

### `test_images`

array of test images 28x28

### `test_labels`

array of labels for each image form test_images

### show_image()

```py
show_image(type, index)
```

  `type` - 'train' or 'test' decides if code takes image from train_images or test_images
  
  `index` - index of image in train_images/test_images (depends on type arg)
  
shows image of given time and index


# comparison

| batches| 3|10|30|100|300|max|
|--------|------|-|-|-|-|-|
|str1| acc: 34% t: 5.95|acc: 54% t: 19|acc: 80% t: 58|acc: 72% t: 203|acc: 85% t: 544|acc: 94% t: 1730|
|str2| acc: 27% t: 5.96|acc: 54% t: 19|acc: 77% t: 57|acc: 80% t: 189|acc: 78% t: 607|acc: 92% t: 1712|