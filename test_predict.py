import sys, numpy as np
sys.path.insert(0, '.')
from backend.services.prediction_service import predict, _model_input_shape, preprocess_image, registry

arr = np.zeros((280, 280, 3), dtype=np.uint8)
arr[:] = 200

model = registry.get('mobilenetv2')
if model:
    h, w, c = _model_input_shape(model)
    print(f'model.input_shape : {model.input_shape}')
    print(f'_model_input_shape: h={h}, w={w}, c={c}')
    tensor = preprocess_image(arr, h, w, c)
    print(f'tensor.shape      : {tensor.shape}')
    result = predict(arr, 'mobilenetv2')
    print(f'predicted_digit   : {result["predicted_digit"]}')
    print(f'confidence        : {result["confidence"]}')
else:
    print('Model not found!')
