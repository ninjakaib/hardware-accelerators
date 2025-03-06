// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[7:0] a;
    input[7:0] b;
    output[7:0] result;

    wire[15:0] tmp8;
    wire[7:0] tmp9;

    // Combinational
    assign result = tmp9;
    assign tmp8 = a * b;
    assign tmp9 = {tmp8[7], tmp8[6], tmp8[5], tmp8[4], tmp8[3], tmp8[2], tmp8[1], tmp8[0]};

endmodule

