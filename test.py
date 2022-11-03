from inference import SegRetino

# Initializing the SegRetino Inference
seg = SegRetino("results/input/input1.png")

# Running inference
seg.inference(set_weight_dir = 'FundusDBNet.pth', path = 'output.png', blend_path = 'blend.png')
