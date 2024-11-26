from pyrtl import *

def accum(size, data_in, waddr, wen, wclear, raddr, lastvec):
    '''A single 32-bit accumulator with 2^size 32-bit buffers.
    On wen, writes data_in to the specified address (waddr) if wclear is high;
    otherwise, it performs an accumulate at the specified address (buffer[waddr] += data_in).
    lastvec is a control signal indicating that the operation being stored now is the
    last vector of a matrix multiply instruction (at the final accumulator, this becomes
    a "done" signal).
    '''

    mem = MemBlock(bitwidth=32, addrwidth=size)
    
    # Writes
    with conditional_assignment:
        with wen:
            with wclear:
                mem[waddr] |= data_in
            with otherwise:
                mem[waddr] |= (data_in + mem[waddr])[:mem.bitwidth]

    # Read
    data_out = mem[raddr]

    # Pipeline registers
    waddrsave = Register(len(waddr))
    waddrsave.next <<= waddr
    wensave = Register(1)
    wensave.next <<= wen
    wclearsave = Register(1)
    wclearsave.next <<= wclear
    lastsave = Register(1)
    lastsave.next <<= lastvec

    return data_out, waddrsave, wensave, wclearsave, lastsave

def accumulators(accsize, datas_in, waddr, we, wclear, raddr, lastvec):
    '''
    Produces array of accumulators of same dimension as datas_in.
    '''

    #probe(we, "accum_wen")
    #probe(wclear, "accum_wclear")
    #probe(waddr, "accum_waddr")

    accout = [ None for i in range(len(datas_in)) ]
    waddrin = waddr
    wein = we
    wclearin = wclear
    lastvecin = lastvec
    for i,x in enumerate(datas_in):
        #probe(x, "acc_{}_in".format(i))
        #probe(wein, "acc_{}_we".format(i))
        #probe(waddrin, "acc_{}_waddr".format(i))
        dout, waddrin, wein, wclearin, lastvecin = accum(accsize, x, waddrin, wein, wclearin, raddr, lastvecin)
        accout[i] = dout
        done = lastvecin

    return accout, done