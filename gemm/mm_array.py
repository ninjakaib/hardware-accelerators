from pyrtl import *
from pyrtl import rtllib
from pyrtl.rtllib import multipliers
from mac import MAC

def MMArray(data_width, matrix_size, data_in, new_weights, weights_in, weights_we):
    '''
    data_in: 256-array of 8-bit activation values from systolic_setup buffer
    new_weights: 256-array of 1-bit control values indicating that new weight should be used
    weights_in: output of weight FIFO (8 x matsize x matsize bit wire)
    weights_we: 1-bit signal to begin writing new weights into the matrix
    '''

    # For signals going to the right, store in a var; for signals going down, keep a list
    # For signals going down, keep a copy of inputs to top row to connect to later
    weights_in_top = [ WireVector(data_width) for i in range(matrix_size) ]  # input weights to top row
    weights_in_last = [x for x in weights_in_top]
    weights_enable_top = [ WireVector(1) for i in range(matrix_size) ]  # weight we to top row
    weights_enable = [x for x in weights_enable_top]
    weights_tag_top = [ WireVector(data_width) for i in range(matrix_size) ]  # weight row tag to top row
    weights_tag = [x for x in weights_tag_top]
    data_out = [Const(0) for i in range(matrix_size)]  # will hold output from final row
    # Build array of MACs
    for i in range(matrix_size):  # for each row
        din = data_in[i]
        switchin = new_weights[i]
        #probe(switchin, "switch" + str(i))
        for j in range(matrix_size):  # for each column
            acc_out, din, switchin, newweight, newwe, newtag = MAC(data_width, matrix_size, din, data_out[j], switchin, weights_in_last[j], weights_enable[j], weights_tag[j])
            #probe(data_out[j], "MACacc{}_{}".format(i, j))
            #probe(acc_out, "MACout{}_{}".format(i, j))
            #probe(din, "MACdata{}_{}".format(i, j))
            weights_in_last[j] = newweight
            weights_enable[j] = newwe
            weights_tag[j] = newtag
            data_out[j] = acc_out
    
    # Handle weight reprogramming
    programming = Register(1)  # when 1, we're in the process of loading new weights
    size = 1
    while pow(2, size) < matrix_size:
        size = size + 1
    progstep = Register(size)  # 256 steps to program new weights (also serves as tag input)
    with conditional_assignment:
        with weights_we & (~programming):
            programming.next |= 1
        with programming & (progstep == matrix_size-1):
            programming.next |= 0
        with otherwise:
            pass
        with programming:  # while programming, increment state each cycle
            progstep.next |= progstep + 1
        with otherwise:
            progstep.next |= Const(0)

    # Divide FIFO output into rows (each row datawidth x matrixsize bits)
    rowsize = data_width * matrix_size
    weight_arr = [ weights_in[i*rowsize : i*rowsize + rowsize] for i in range(matrix_size) ]
    # Mux the wire for this row
    current_weights_wire = mux(progstep, *weight_arr)
    # Split the wire into an array of 8-bit values
    current_weights = [ current_weights_wire[i*data_width:i*data_width+data_width] for i in reversed(range(matrix_size)) ]

    # Connect top row to input and control signals
    for i, win in enumerate(weights_in_top):
        # From the current 256-array, select the byte for this column
        win <<= current_weights[i]
    for we in weights_enable_top:
        # Whole row gets same signal: high when programming new weights
        we <<= programming
    for wt in weights_tag_top:
        # Tag is same for whole row; use state index (runs from 0 to 255)
        wt <<= progstep

    return [ x.sign_extended(32) for x in data_out ]

