import onnxruntime as ort

# Defina as opções da sessão
sess_options = ort.SessionOptions()

# Especifique o provedor CUDA
# providers = [('CUDAExecutionProvider', {
#     'device_id': 0,
# })]
# Especifique os provedores sem o TensorrtExecutionProvider
providers = ['CPUExecutionProvider']
# Tente criar uma sessão de inferência
try:
    sess = ort.InferenceSession('dla34_Opset16.onnx', sess_options=sess_options, providers=providers)
    print("CUDA Execution Provider carregado com sucesso!")
except Exception as e:
    print("Erro ao carregar o CUDA Execution Provider:")
    print(e)
