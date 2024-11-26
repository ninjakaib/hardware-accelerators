from pyrtl import *
from pyrtl import rtllib
from pyrtl.rtllib import multipliers

#set_debug_mode()
globali = 0  # To give unique numbers to each MAC
def MAC(data_width, matrix_size, data_in, acc_in, switchw, weight_in, weight_we, weight_tag):
    '''Multiply-Accumulate unit with programmable weight.
    Inputs
    data_in: The 8-bit activation value to multiply by weight.
    acc_in: 32-bit value to accumulate with product.
    switchw: Control signal; when 1, switch to using the other weight buffer.
    weight_in: 8-bit value to write to the secondary weight buffer.
    weight_we: When high, weights are being written; if tag matches, store weights.
               Otherwise, pass them through with incremented tag.
    weight_tag: If equal to 255, weight is for this row; store it.

    Outputs
    out: Result of the multiply accumulate; moves one cell down to become acc_in.
    data_reg: data_in, stored in a pipeline register for cell to the right.
    switch_reg: switchw, stored in a pipeline register for cell to the right.
    weight_reg: weight_in, stored in a pipeline register for cell below.
    weight_we_reg: weight_we, stored in a pipeline register for cell below.
    weight_tag_reg: weight_tag, incremented and stored in a pipeline register for cell below
    '''
    global globali
    # Check lengths of inupts
    if len(weight_in) != len(data_in) != data_width:
        raise Exception("Expected 8-bit value in MAC.")
    if len(switchw) != len(weight_we) != 1:
        raise Exception("Expected 1-bit control signal in MAC.")

    # Should never switch weight buffers while they're changing
    #rtl_assert(~(weight_we & switchw), Exception("Cannot switch weight values when they're being loaded!"))

    # Use two buffers to store weight and next weight to use.
    wbuf1, wbuf2 = Register(len(weight_in)), Register(len(weight_in))

    # Track which buffer is current and which is secondary.
    current_buffer_reg = Register(1)
    with conditional_assignment:
        with switchw:
            current_buffer_reg.next |= ~current_buffer_reg
    current_buffer = current_buffer_reg ^ switchw  # reflects change in same cycle switchw goes high

    # When told, store a new weight value in the secondary buffer
    with conditional_assignment:
        with weight_we & (weight_tag == Const(matrix_size-1)):
            with current_buffer == 0:  # If 0, wbuf1 is current; if 1, wbuf2 is current
                wbuf2.next |= weight_in
            with otherwise:
                wbuf1.next |= weight_in

    # Do the actual MAC operation
    weight = select(current_buffer, wbuf2, wbuf1)
    #probe(weight, "weight" + str(globali))
    globali += 1
    #inlen = max(len(weight), len(data_in))
    #product = weight.sign_extended(inlen*2) * data_in.sign_extended(inlen*2)
    #product = product[:inlen*2]
    product = helperfuncs.mult_signed(weight, data_in)[:32]
    #plen = len(weight) + len(data_in)
    #product = weight.sign_extended(plen) * data_in.sign_extended(plen)
    #product = product[:plen]
    l = max(len(product), len(acc_in)) + 1
    out = (product.sign_extended(l) + acc_in.sign_extended(l))[:-1]

    #product = rtllib.multipliers.signed_tree_multiplier(weight, data_in)
    #l = max(len(product), len(acc_in))
    #out = product.sign_extended(l) + acc_in.sign_extended(l)

    if len(out) > 32:
        out = out[:32]
                
    # For values that need to be forward to the right/bottom, store in pipeline registers
    data_reg = Register(len(data_in))  # pipeline register, holds data value for cell to the right
    data_reg.next <<= data_in
    switch_reg = Register(1)  # pipeline register, holds switch control signal for cell to the right
    switch_reg.next <<= switchw
    acc_reg = Register(len(out))  # output value for MAC below
    acc_reg.next <<= out
    weight_reg = Register(len(weight_in))  # pipeline register, holds weight input for cell below
    weight_reg.next <<= weight_in
    weight_we_reg = Register(1)  # pipeline register, holds weight write enable signal for cell below
    weight_we_reg.next <<= weight_we
    weight_tag_reg = Register(len(weight_tag))  # pipeline register, holds weight tag for cell below
    weight_tag_reg.next <<= (weight_tag + 1)[:len(weight_tag)]  # increment tag as it passes down rows

    return acc_reg, data_reg, switch_reg, weight_reg, weight_we_reg, weight_tag_reg
