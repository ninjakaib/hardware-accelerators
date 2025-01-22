from typing import Type, Union, Dict, Optional
import numpy as np
from .base import BaseFloat, FormatSpec

def convert_float_type(value: BaseFloat, target_type: Type[BaseFloat]) -> BaseFloat:
    """
    Convert between different floating point types.
    This performs a direct bit-level conversion by truncating or extending the mantissa.
    
    Args:
        value: Source floating point value
        target_type: Target floating point type class
        
    Returns:
        Converted value in target type
    """
    # Instead of bit manipulation, we'll use the actual float value for conversion
    # This ensures proper handling of exponent bias differences
    float_val = float(value)
    return target_type(float_val)

def quantize_weights(weights: np.ndarray, 
                    target_type: Type[BaseFloat], 
                    scale_factor: Optional[float] = None,
                    per_channel: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Quantize weights to target floating point type with optional scaling.
    
    Args:
        weights: Input weight tensor
        target_type: Target floating point type class
        scale_factor: Optional scaling factor. If None, will be computed automatically
        per_channel: Whether to compute scale factors per output channel
        
    Returns:
        If per_channel is False and scale_factor is None:
            Quantized weights array
        Otherwise:
            Dict containing:
                'weights': Quantized weights array
                'scales': Scale factors per channel (if per_channel=True)
                'scale': Single scale factor (if per_channel=False and scale_factor is not None)
    """
    # Create temporary instance to get format spec
    temp_target = target_type(0.0)
    format_spec = temp_target.format_spec
    
    def quantize_array(arr, scale):
        """Helper function to quantize an array with a given scale factor"""
        scaled = arr / scale
        # Vectorize the quantization operation
        return np.array([float(target_type(float(x))) for x in scaled.flat]).reshape(scaled.shape)
    
    if per_channel:
        # Compute per-channel scale factors
        if scale_factor is not None:
            raise ValueError("Cannot specify scale_factor when per_channel=True")
            
        # For 2D weight matrices: (output_features, input_features)
        # For 4D conv weights: (output_channels, input_channels, height, width)
        # We want to scale per output channel/feature, so use axis=1 for remaining dims
        if weights.ndim == 4:
            max_vals = np.max(np.abs(weights), axis=(1, 2, 3))  # Changed from axis=(0,2,3)
        else:
            max_vals = np.max(np.abs(weights), axis=1)  # Changed from axis=0
            
        scales = max_vals / format_spec.max_normal
        
        # Quantize each channel
        quantized = np.zeros_like(weights)
        if weights.ndim == 4:
            for i in range(weights.shape[0]):  # Changed from shape[1]
                quantized[i] = quantize_array(weights[i], scales[i])
        else:
            for i in range(weights.shape[0]):  # Changed from shape[1]
                quantized[i] = quantize_array(weights[i], scales[i])
            
        return {
            'weights': quantized,
            'scales': scales
        }
    else:
        # Use single scale factor
        if scale_factor is None:
            max_val = np.max(np.abs(weights))
            scale_factor = max_val / format_spec.max_normal
            
        # Scale and quantize
        quantized = quantize_array(weights, scale_factor)
        
        # Return just the array if no scaling info needed
        if scale_factor is None:
            return quantized
            
        return {
            'weights': quantized,
            'scale': scale_factor
        } 