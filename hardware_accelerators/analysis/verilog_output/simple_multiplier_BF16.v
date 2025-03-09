// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[15:0] a;
    input[15:0] b;
    output[15:0] result;

    wire[31:0] tmp12;
    wire[15:0] tmp13;

    // Combinational
    assign result = tmp13;
    assign tmp12 = a * b;
    assign tmp13 = {tmp12[15], tmp12[14], tmp12[13], tmp12[12], tmp12[11], tmp12[10], tmp12[9], tmp12[8], tmp12[7], tmp12[6], tmp12[5], tmp12[4], tmp12[3], tmp12[2], tmp12[1], tmp12[0]};

endmodule

