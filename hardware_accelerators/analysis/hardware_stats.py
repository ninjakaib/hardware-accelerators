import pandas as pd
import numpy as np
from typing import Literal, Dict, Tuple, List, Any, Optional

# Mapping from UI options to dataframe values
DTYPE_MAP = {"float8": "fp8", "bfloat16": "bf16", "float32": "fp32"}

SPEED_MAP = {"Fast": True, "Efficient": False}

PIPELINE_MAP = {
    "None": "combinational",
    "Low": "combinational",
    "Full": "pipelined",
    "High": "pipelined",
}


def filter_components(
    df: pd.DataFrame, operation: str, dtype: str, is_fast: bool, architecture: str
) -> pd.DataFrame:
    """
    Filter the dataframe to get components matching the specified criteria.

    Args:
        df: DataFrame containing component data
        operation: Type of operation ('multiplier', 'lmul', 'adder')
        dtype: Data type ('fp8', 'bf16', 'fp32')
        is_fast: Whether to use fast components
        architecture: Architecture type ('combinational', 'pipelined')

    Returns:
        Filtered DataFrame
    """
    filtered_df = df[
        (df["operation"] == operation)
        & (df["dtype"] == dtype)
        & (df["is_fast"] == is_fast)
        & (df["architecture"] == architecture)
    ]

    if filtered_df.empty:
        # If no exact match, try without the is_fast constraint
        filtered_df = df[
            (df["operation"] == operation)
            & (df["dtype"] == dtype)
            & (df["architecture"] == architecture)
        ]

    if filtered_df.empty:
        # If still no match, try without architecture constraint
        filtered_df = df[(df["operation"] == operation) & (df["dtype"] == dtype)]

    return filtered_df


def get_pe_components(
    df: pd.DataFrame, mult_type: str, dtype: str, is_fast: bool, architecture: str
) -> Tuple[pd.Series, pd.Series]:
    """
    Get the multiplier/lmul and adder components for a PE.

    Args:
        df: DataFrame containing component data
        mult_type: Type of multiplier ('multiplier' or 'lmul')
        dtype: Data type ('fp8', 'bf16', 'fp32')
        is_fast: Whether to use fast components
        architecture: Architecture type ('combinational', 'pipelined')

    Returns:
        Tuple of (multiplier, adder) Series
    """
    mult_df = filter_components(df, mult_type, dtype, is_fast, architecture)
    adder_df = filter_components(df, "adder", dtype, is_fast, architecture)

    if mult_df.empty or adder_df.empty:
        raise ValueError(
            f"Could not find components for {mult_type}, {dtype}, {is_fast}, {architecture}"
        )

    # Get the first matching component
    multiplier = mult_df.iloc[0]
    adder = adder_df.iloc[0]

    return multiplier, adder


def calculate_pe_metrics(multiplier: pd.Series, adder: pd.Series) -> Dict[str, float]:
    """
    Calculate metrics for a single PE (Processing Element).

    Args:
        multiplier: Series containing multiplier component data
        adder: Series containing adder component data

    Returns:
        Dictionary of PE metrics
    """
    # Area and power are additive
    area = multiplier["area"] + adder["area"]
    power = multiplier["power"] + adder["power"]

    # Critical path depends on architecture
    if (
        multiplier["architecture"] == "combinational"
        and adder["architecture"] == "combinational"
    ):
        delay = max(multiplier["max_arrival_time"], adder["max_arrival_time"])
        pipeline_stages = 1
    else:
        # For pipelined designs, we assume the critical path is the slowest stage
        delay = max(multiplier["max_arrival_time"], adder["max_arrival_time"])
        # Estimate pipeline stages
        if multiplier["architecture"] == "pipelined":
            mult_stages = 4  # Assumption for pipelined multiplier
        else:
            mult_stages = 1

        if adder["architecture"] == "pipelined":
            add_stages = 5  # Assumption for pipelined adder
        else:
            add_stages = 1

        pipeline_stages = mult_stages + add_stages - 1  # -1 because they share a stage

    # Calculate performance metrics
    clock_freq_ghz = 1.0 / delay  # GHz, assuming delay is in ns
    ops_per_cycle = 2  # 1 multiply + 1 add = 2 FLOPs per cycle
    tflops = clock_freq_ghz * ops_per_cycle / 1000  # TFLOPS for a single PE

    # Efficiency metrics
    tflops_per_watt = tflops / power
    tflops_per_mm2 = tflops / (area * 1e-6)  # Convert area to mm²
    power_density = power / (area * 1e-6)  # W/mm²

    return {
        "area_um2": area,
        "area_mm2": area * 1e-6,
        "power_w": power,
        "delay_ns": delay,
        "clock_freq_ghz": clock_freq_ghz,
        "pipeline_stages": pipeline_stages,
        "tflops": tflops,
        "tflops_per_watt": tflops_per_watt,
        "tflops_per_mm2": tflops_per_mm2,
        "power_density": power_density,
        "energy_per_op_pj": (power / (clock_freq_ghz * ops_per_cycle * 1e9))
        * 1e12,  # pJ per operation
    }


def calculate_array_metrics(
    pe_metrics: Dict[str, float], array_size: int, num_cores: int
) -> Dict[str, float]:
    """
    Calculate metrics for a systolic array and the entire accelerator.

    Args:
        pe_metrics: Dictionary of PE metrics
        array_size: Size of the systolic array (NxN)
        num_cores: Number of accelerator cores

    Returns:
        Dictionary of array and accelerator metrics
    """
    # Number of PEs per array and total
    pes_per_array = array_size * array_size
    total_pes = pes_per_array * num_cores

    # Scale metrics for a single array
    array_area_mm2 = pe_metrics["area_mm2"] * pes_per_array
    array_power_w = pe_metrics["power_w"] * pes_per_array

    # Scale metrics for the entire accelerator
    total_area_mm2 = array_area_mm2 * num_cores
    total_power_w = array_power_w * num_cores

    # Performance scales with the number of PEs
    array_tflops = pe_metrics["tflops"] * pes_per_array
    total_tflops = array_tflops * num_cores

    # Efficiency metrics
    total_tflops_per_watt = total_tflops / total_power_w
    total_tflops_per_mm2 = total_tflops / total_area_mm2

    # Latency calculation
    # For an NxN array, data takes 2N-1 cycles to flow through
    # Plus pipeline_stages-1 cycles for the pipeline to fill
    pipeline_latency_cycles = pe_metrics["pipeline_stages"] - 1
    array_latency_cycles = 2 * array_size - 1
    total_latency_cycles = array_latency_cycles + pipeline_latency_cycles

    # Time per cycle based on clock frequency
    cycle_time_ns = 1.0 / pe_metrics["clock_freq_ghz"]

    # Total latency in ns
    total_latency_ns = total_latency_cycles * cycle_time_ns

    # Throughput after pipeline is filled (ops per second)
    throughput_ops_per_second = (
        pe_metrics["clock_freq_ghz"] * 1e9 * pes_per_array * 2
    )  # 2 ops per PE per cycle
    total_throughput_ops_per_second = throughput_ops_per_second * num_cores

    # Energy per matrix multiplication
    # Assuming an NxN matrix multiply requires N³ operations
    ops_per_matmul = array_size**3
    energy_per_matmul_nj = (
        (array_power_w / throughput_ops_per_second) * ops_per_matmul * 1e9
    )  # nJ

    # Inference metrics (assuming a simple MLP with 3 layers)
    # Each layer requires a matrix multiplication
    num_layers = 3
    inference_ops = ops_per_matmul * num_layers
    inference_latency_ns = (inference_ops / throughput_ops_per_second) * 1e9
    inference_energy_uj = (
        (total_power_w / total_throughput_ops_per_second) * inference_ops * 1e6
    )  # uJ

    return {
        "array_size": array_size,
        "num_cores": num_cores,
        "pes_per_array": pes_per_array,
        "total_pes": total_pes,
        "clock_freq_ghz": pe_metrics["clock_freq_ghz"],
        "array_area_mm2": array_area_mm2,
        "total_area_mm2": total_area_mm2,
        "array_power_w": array_power_w,
        "total_power_w": total_power_w,
        "array_tflops": array_tflops,
        "total_tflops": total_tflops,
        "tflops_per_watt": total_tflops_per_watt,
        "tflops_per_mm2": total_tflops_per_mm2,
        "power_density_w_mm2": total_power_w / total_area_mm2,
        "total_latency_cycles": total_latency_cycles,
        "total_latency_ns": total_latency_ns,
        "throughput_gops": total_throughput_ops_per_second / 1e9,  # GOPS
        "energy_per_matmul_nj": energy_per_matmul_nj,
        "inference_latency_ns": inference_latency_ns,
        "inference_latency_us": inference_latency_ns / 1e3,  # us
        "inference_energy_uj": inference_energy_uj,
        "inferences_per_second": 1e9 / inference_latency_ns,
        "inferences_per_watt": (1e9 / inference_latency_ns) / total_power_w,
    }


def format_metrics_for_display(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format metrics for display in the Gradio UI.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Dictionary of formatted metrics
    """
    formatted = {}

    # Format area
    formatted["Total Chip Area"] = f"{metrics['total_area_mm2']:.2f} mm²"

    # Format performance
    formatted["Clock Speed"] = f"{metrics['clock_freq_ghz']:.2f} GHz"
    formatted["Total Performance"] = f"{metrics['total_tflops']:.2f} TFLOPS"
    formatted["Performance per Core"] = f"{metrics['array_tflops']:.2f} TFLOPS"
    formatted["Performance per Watt"] = f"{metrics['tflops_per_watt']:.2f} TFLOPS/W"
    formatted["Performance per Area"] = f"{metrics['tflops_per_mm2']:.2f} TFLOPS/mm²"

    # Format power
    formatted["Total Power"] = f"{metrics['total_power_w']:.2f} W"
    formatted["Power Density"] = f"{metrics['power_density_w_mm2']:.2f} W/mm²"

    # Format latency and throughput
    formatted["Matrix Mult Latency"] = f"{metrics['total_latency_ns']:.2f} ns"
    formatted["Inference Latency"] = f"{metrics['inference_latency_us']:.2f} µs"
    formatted["Throughput"] = f"{metrics['throughput_gops']:.2f} GOPS"

    # Format energy
    formatted["Energy per Matrix Mult"] = f"{metrics['energy_per_matmul_nj']:.2f} nJ"
    # formatted["Inference Energy"] = f"{metrics['inference_energy_uj']:.2f} µJ"
    # formatted["Inferences per Second"] = f"{metrics['inferences_per_second']:.0f}"
    # formatted["Inferences per Watt"] = f"{metrics['inferences_per_watt']:.0f}"

    return formatted


def calculate_hardware_stats(
    df: pd.DataFrame,
    activation_type: Literal["float8", "bfloat16", "float32"],
    weight_type: Literal["float8", "bfloat16", "float32"],
    systolic_array_size: int,
    num_accelerator_cores: int,
    fast_internals: Literal["Fast", "Efficient"],
    pipeline_level: Literal["None", "Low", "Full"],
    process_node_size: Optional[Literal["7nm", "45nm", "130nm"]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Calculate hardware statistics for both lmul and standard IEEE multiplier configurations.

    Args:
        df: DataFrame containing component data
        activation_type: Type of activations
        weight_type: Type of weights
        systolic_array_size: Size of the systolic array
        num_accelerator_cores: Number of accelerator cores
        fast_internals: Whether to use fast or efficient components
        pipeline_level: Level of pipelining
        process_node_size: Process node size (ignored for now)

    Returns:
        Tuple of (lmul_metrics, ieee_metrics) dictionaries
    """
    # Map UI options to dataframe values
    act_dtype = DTYPE_MAP[activation_type]
    weight_dtype = DTYPE_MAP[weight_type]
    is_fast = SPEED_MAP[fast_internals]
    architecture = PIPELINE_MAP[pipeline_level]

    # For mixed precision, use the larger precision for the PE
    pe_dtype = (
        act_dtype
        if DTYPE_MAP[activation_type] >= DTYPE_MAP[weight_type]
        else weight_dtype
    )

    # Calculate metrics for lmul configuration
    try:
        lmul_mult, lmul_adder = get_pe_components(
            df, "lmul", pe_dtype, is_fast, architecture
        )
        lmul_pe_metrics = calculate_pe_metrics(lmul_mult, lmul_adder)
        lmul_array_metrics = calculate_array_metrics(
            lmul_pe_metrics, systolic_array_size, num_accelerator_cores
        )
        lmul_formatted = format_metrics_for_display(lmul_array_metrics)
    except ValueError as e:
        # If lmul components not found, return error message
        lmul_formatted = {"Error": f"Could not find lmul components: {str(e)}"}

    # Calculate metrics for standard IEEE multiplier configuration
    try:
        ieee_mult, ieee_adder = get_pe_components(
            df, "multiplier", pe_dtype, is_fast, architecture
        )
        ieee_pe_metrics = calculate_pe_metrics(ieee_mult, ieee_adder)
        ieee_array_metrics = calculate_array_metrics(
            ieee_pe_metrics, systolic_array_size, num_accelerator_cores
        )
        ieee_formatted = format_metrics_for_display(ieee_array_metrics)
    except ValueError as e:
        # If IEEE components not found, return error message
        ieee_formatted = {"Error": f"Could not find IEEE components: {str(e)}"}

    return lmul_formatted, ieee_formatted


def calculate_comparison_metrics(
    lmul_metrics: Dict[str, str], ieee_metrics: Dict[str, str]
) -> Dict[str, str]:
    """
    Calculate comparison metrics between lmul and IEEE configurations.

    Args:
        lmul_metrics: Dictionary of lmul metrics
        ieee_metrics: Dictionary of IEEE metrics

    Returns:
        Dictionary of comparison metrics
    """
    comparison = {}

    # Check if there was an error in either calculation
    if "Error" in lmul_metrics or "Error" in ieee_metrics:
        return {"Error": "Cannot calculate comparison due to missing components"}

    # Extract numeric values from formatted strings
    def extract_number(s):
        return float(s.split()[0])

    # Calculate percentage improvements
    try:
        # Area improvement (lower is better)
        lmul_area = extract_number(lmul_metrics["Total Chip Area"])
        ieee_area = extract_number(ieee_metrics["Total Chip Area"])
        area_improvement = (1 - lmul_area / ieee_area) * 100
        comparison["Area Reduction"] = f"{area_improvement:.1f}%"

        # Performance improvement (higher is better)
        lmul_perf = extract_number(lmul_metrics["Total Performance"])
        ieee_perf = extract_number(ieee_metrics["Total Performance"])
        perf_improvement = (lmul_perf / ieee_perf - 1) * 100
        comparison["Performance Improvement"] = f"{perf_improvement:.1f}%"

        # Power efficiency improvement (higher is better)
        lmul_eff = extract_number(lmul_metrics["Performance per Watt"])
        ieee_eff = extract_number(ieee_metrics["Performance per Watt"])
        eff_improvement = (lmul_eff / ieee_eff - 1) * 100
        comparison["Efficiency Improvement"] = f"{eff_improvement:.1f}%"

        # Latency improvement (lower is better)
        lmul_latency = extract_number(lmul_metrics["Inference Latency"])
        ieee_latency = extract_number(ieee_metrics["Inference Latency"])
        latency_improvement = (1 - lmul_latency / ieee_latency) * 100
        comparison["Latency Reduction"] = f"{latency_improvement:.1f}%"

        # Energy improvement (lower is better)
        lmul_energy = extract_number(lmul_metrics["Inference Energy"])
        ieee_energy = extract_number(ieee_metrics["Inference Energy"])
        energy_improvement = (1 - lmul_energy / ieee_energy) * 100
        comparison["Energy Reduction"] = f"{energy_improvement:.1f}%"

    except (ValueError, KeyError) as e:
        comparison["Error"] = f"Error calculating comparisons: {str(e)}"

    return comparison


# Example usage in the Gradio app:
def update_hardware_stats(
    df: pd.DataFrame,
    activation_type: str,
    weight_type: str,
    systolic_array_size: int,
    num_accelerator_cores: int,
    fast_internals: str,
    pipeline_level: str,
    process_node_size: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Update hardware statistics for the Gradio app.

    Args:
        df: DataFrame containing component data
        activation_type: Type of activations
        weight_type: Type of weights
        systolic_array_size: Size of the systolic array
        num_accelerator_cores: Number of accelerator cores
        fast_internals: Whether to use fast or efficient components
        pipeline_level: Level of pipelining
        process_node_size: Process node size

    Returns:
        Tuple of (lmul_metrics, ieee_metrics, comparison_metrics) dictionaries
    """
    lmul_metrics, ieee_metrics = calculate_hardware_stats(
        df,
        activation_type,
        weight_type,
        systolic_array_size,
        num_accelerator_cores,
        fast_internals,
        pipeline_level,
        process_node_size,
    )

    comparison_metrics = calculate_comparison_metrics(lmul_metrics, ieee_metrics)

    return lmul_metrics, ieee_metrics, comparison_metrics
