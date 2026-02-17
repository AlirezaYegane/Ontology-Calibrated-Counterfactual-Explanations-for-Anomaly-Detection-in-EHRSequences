import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
    x = torch.randn((2048, 2048), device="cuda")
    y = torch.randn((2048, 2048), device="cuda")
    z = (x @ y).mean()
    torch.cuda.synchronize()
    print("GPU matmul ok, mean:", float(z))
else:
    print("gpu_name: CPU")
    x = torch.randn((512, 512))
    y = torch.randn((512, 512))
    z = (x @ y).mean()
    print("CPU matmul ok, mean:", float(z))
