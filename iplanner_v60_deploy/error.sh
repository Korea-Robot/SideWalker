[ERROR] [1761136046.543297471] [semantic_pointcloud_node]: 처리 오류: Unexpected type <class 'numpy.ndarray'>
[ERROR] [1761136046.544140982] [semantic_pointcloud_node]: Traceback (most recent call last):
  File "/home/krm/SideWalker/iplanner_v60_deploy/semantic_point_node.py", line 279, in synchronized_callback
    semantic_mask = self.run_segmentation(rgb_image)
  File "/home/krm/SideWalker/iplanner_v60_deploy/semantic_point_node.py", line 362, in run_segmentation
    input_tensor = self.transform(rgb_image_rgb)
  File "/home/krm/.local/lib/python3.10/site-packages/torchvision/transforms/transforms.py", line 95, in __call__
    img = t(img)
  File "/home/krm/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/krm/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/krm/.local/lib/python3.10/site-packages/torchvision/transforms/transforms.py", line 354, in forward
    return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
  File "/home/krm/.local/lib/python3.10/site-packages/torchvision/transforms/functional.py", line 456, in resize
    _, image_height, image_width = get_dimensions(img)
  File "/home/krm/.local/lib/python3.10/site-packages/torchvision/transforms/functional.py", line 80, in get_dimensions
    return F_pil.get_dimensions(img)
  File "/home/krm/.local/lib/python3.10/site-packages/torchvision/transforms/_functional_pil.py", line 31, in get_dimensions
    raise TypeError(f"Unexpected type {type(img)}")
TypeError: Unexpected type <class 'numpy.ndarray'>