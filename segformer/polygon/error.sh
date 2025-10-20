krm@ubuntu:~/segformer/polygon$ python3 polygon_segformer_inference.py 
/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.26.1
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
1756297910.648494 [123] pt_main_th: config: //CycloneDDS/Domain/General: 'NetworkInterfaceAddress': deprecated element (file:///home/krm/.cyclonedds.xml line 8)
[INFO] [1756297910.718256082] [realsense_inference_node]: Realsense Inference node has started.
Traceback (most recent call last):
  File "/home/krm/segformer/polygon/polygon_segformer_inference.py", line 174, in <module>
    main()
  File "/home/krm/segformer/polygon/polygon_segformer_inference.py", line 160, in main
    node = SemanticInferenceNode()
  File "/home/krm/segformer/polygon/polygon_segformer_inference.py", line 67, in __init__
    self.model = self.load_model()
  File "/home/krm/segformer/polygon/polygon_segformer_inference.py", line 88, in load_model
    model = SegformerForSemanticSegmentation.from_pretrained(
  File "/home/krm/.local/lib/python3.10/site-packages/transformers/modeling_utils.py", line 317, in _wrapper
    return func(*args, **kwargs)
  File "/home/krm/.local/lib/python3.10/site-packages/transformers/modeling_utils.py", line 5074, in from_pretrained
    ) = cls._load_pretrained_model(
  File "/home/krm/.local/lib/python3.10/site-packages/transformers/modeling_utils.py", line 5340, in _load_pretrained_model
    load_state_dict(checkpoint_files[0], map_location="meta", weights_only=weights_only).keys()
  File "/home/krm/.local/lib/python3.10/site-packages/transformers/modeling_utils.py", line 562, in load_state_dict
    check_torch_load_is_safe()
  File "/home/krm/.local/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1622, in check_torch_load_is_safe
    raise ValueError(
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434

