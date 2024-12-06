import pyrtl
from enum import IntEnum

"""
Register instructions:
op rd rs2 rs1 : rd <- (op rs1 rs2)
  with op in (ADD|SUB|XOR|AND|OR)

Unconditional branch:
JUMP immediate : pc <- (add pc immediate)
"""

WIDTH = 16

class OP(IntEnum):
    NOP  = 0x0 # Unused
    ADD  = 0x1
    SUB  = 0x2
    XOR  = 0x3
    AND  = 0x4
    OR   = 0x5
    JUMP = 0x6

# Declarations:
# -------------

# Stall input - if high, propagate NOP through pipeline
stall = pyrtl.Input(1, 'stall')

# Program counter
pc = pyrtl.Register(WIDTH, 'pc')

# Instruction memory
imem = pyrtl.MemBlock(bitwidth=WIDTH, addrwidth=WIDTH, name='imem')

# Register file is a memory mapping 4-bit address to 16-bit data value
# Zero register is reserved 0 value
rf = pyrtl.MemBlock(bitwidth=WIDTH, addrwidth=4, name='rf', asynchronous=True)

# Pipeline registers:
#   `d_` and `x_` indicate which stage the registers are used
#   (`d` for decode, `x` for execute)
d_inst = pyrtl.Register(WIDTH, 'd_inst')
d_pc = pyrtl.Register(WIDTH, 'd_pc')
x_rd = pyrtl.Register(4, 'x_rd')
x_ctrl_alu_op = pyrtl.Register(4, 'x_ctrl_alu_op')
x_ctrl_branch = pyrtl.Register(1, 'x_ctrl_branch')
x_ctrl_write = pyrtl.Register(1, 'x_ctrl_write')
x_alu_in1 = pyrtl.Register(WIDTH, 'x_alu_in1')
x_alu_in2 = pyrtl.Register(WIDTH, 'x_alu_in2')

# Wire to hold ALU result
alu_result = pyrtl.WireVector(WIDTH, 'alu_out')

# Stage 1: Fetch
# --------------
branch_taken = pyrtl.WireVector(1, 'branch_taken')
branch_target = pyrtl.WireVector(WIDTH, 'branch_target')

# Read from instruction memory at current PC
# Store fetched instruction in instruction register
# Will be available in the *next* cycle
with pyrtl.conditional_assignment:
    with branch_taken:
        d_inst.next |= 0x1000 # NOP
    with pyrtl.otherwise:
        d_inst.next |= imem[pc]

# Increment PC
# `|=` is PyRTL's "conditional assignment" operator, used in `with` expressions
with pyrtl.conditional_assignment:
    with stall:
        pc.next |= pc
    with branch_taken:
        pc.next |= branch_target
    with pyrtl.otherwise:
        pc.next |= pc + 1

d_pc.next <<= pc

# Stage 2: Decode
# ---------------
nop = pyrtl.WireVector(1, 'nop')
nop <<= branch_taken | stall

# Intermediate wires for op, 2 source registers (a, b), destination register (c)
# A NOP instruction is ADD r0 r0 r0
inst_a = pyrtl.mux(nop, d_inst[0:4], pyrtl.Const(0, bitwidth=4))
inst_b = pyrtl.mux(nop, d_inst[4:8], pyrtl.Const(0, bitwidth=4))
inst_c = pyrtl.mux(nop, d_inst[8:12], pyrtl.Const(0, bitwidth=4))
inst_op = pyrtl.mux(nop, d_inst[12:16], OP.ADD)
inst_imm = d_inst[0:12].zero_extended(WIDTH)

# Control signals
ctrl_branch = pyrtl.WireVector(1, 'ctrl_branch')
ctrl_alu_op = pyrtl.WireVector(4, 'ctrl_alu_op')
ctrl_write = pyrtl.WireVector(1, 'ctrl_write')

# Control logic
# Note: We replace this definition with a "hole" in our synthesis scenario.
ctrl_alu_op <<= pyrtl.enum_mux(
        inst_op,
        {
            OP.JUMP: OP.ADD,
        },
        default=inst_op)

# Note: We replace this definition with a "hole" in our synthesis scenario.
ctrl_branch <<= pyrtl.enum_mux(
        inst_op,
        {
            OP.JUMP: 1,
        },
        default=0)

# Note: We replace this definition with a "hole" in our synthesis scenario.
ctrl_write <<= pyrtl.enum_mux(
        inst_op,
        {
            OP.ADD: 1,
            OP.SUB: 1,
            OP.XOR: 1,
            OP.OR:  1,
            OP.AND: 1,
        },
        default=0)

# Read from register file and do register forwarding
# ALU input 1:
with pyrtl.conditional_assignment:
    with ctrl_branch:
        x_alu_in1.next |= d_pc
    with (x_rd == inst_a) & (inst_a != 0):
        x_alu_in1.next |= alu_result
    with pyrtl.otherwise:
        x_alu_in1.next |= rf[inst_a]

# ALU input 2:
with pyrtl.conditional_assignment:
    with ctrl_branch:
        x_alu_in2.next |= inst_imm
    with (x_rd == inst_b) & (inst_b != 0):
        x_alu_in2.next |= alu_result
    with pyrtl.otherwise:
        x_alu_in2.next |= rf[inst_b]

# Forward control signals to next stage
x_ctrl_alu_op.next <<= ctrl_alu_op
x_ctrl_branch.next <<= ctrl_branch
x_ctrl_write.next <<= ctrl_write
x_rd.next <<= inst_c

# Stage 3: Execute and write back
# -------------------------------
# Perform ALU operation
alu_result <<= pyrtl.enum_mux(
        x_ctrl_alu_op,
        {
            OP.ADD: x_alu_in1 + x_alu_in2,
            OP.SUB: x_alu_in1 - x_alu_in2,
            OP.XOR: x_alu_in1 ^ x_alu_in2,
            OP.OR:  x_alu_in1 | x_alu_in2,
            OP.AND: x_alu_in1 & x_alu_in2,
        },
        default=0)

branch_taken <<= x_ctrl_branch
branch_target <<= alu_result

# Write result back to register file
reg_write_enable = pyrtl.WireVector(1, 'reg_write_enable')
reg_write_enable <<= x_ctrl_write & (x_rd != 0)
rf[x_rd] <<= pyrtl.MemBlock.EnabledWrite(alu_result, reg_write_enable)

# -------------------------------------------------------------------------------

# Start a simulation
sim_trace = pyrtl.SimulationTrace()

# Initialize reg file
rf_init = {}
for i in range(16):
    if i in range(4):
        rf_init[i] = i
    else:
        rf_init[i] = 0

# Load a program
imem_init = {
    0: 0x1112, # ADD r1 r2 r1
    1: 0x1333, # ADD r3 r3 r3
    2: 0x6007, # JUMP 0x7
    3: 0x2221, # SUB r2 r1 r2
    4: 0x1000, # ADD r0 r0 r0
    9: 0x2121, # SUB r1 r1 r2
}

# Initialize simulation with instructions and reg file
sim = pyrtl.Simulation(tracer=sim_trace, memory_value_map={
    rf : rf_init,
    imem : imem_init,
})

# Run
sim.step({'stall': 0})
sim.step({'stall': 0})
sim.step({'stall': 0})
sim.step({'stall': 0})
sim.step({'stall': 0})
sim.step({'stall': 0})
sim.step({'stall': 0})
sim.step({'stall': 1})
sim.step({'stall': 1})

sim_trace.render_trace(symbol_len=10, segment_size=1)

# Print out the register file
print("Reg file contents:")
print(sim.inspect_mem(rf))

# Below prints a "netlist" representation of the design.
# Essentially a directed graph with nodes as gates (operations over wire vectors)
# and edges as wires connecting them.
# ---
# pyrtl.optimize()
# print("Printing netlist...")
# print(pyrtl.working_block())

with open("kai/output/diagrams/pipelined-alu.tgf", 'w') as f:
    pyrtl.output_to_trivialgraph(f, split_state=False)
with open("kai/output/diagrams/pipelined-alu.svg", 'w') as f:
    pyrtl.output_to_svg(f, split_state=False)
