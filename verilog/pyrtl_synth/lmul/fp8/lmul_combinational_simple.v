// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out4019);
    input clk;
    input rst;
    input[7:0] float_a;
    input[7:0] float_b;
    output[7:0] out4019;

    wire[6:0] const_758_55;
    wire const_759_0;
    wire tmp4010;
    wire tmp4011;
    wire tmp4012;
    wire[6:0] tmp4013;
    wire[6:0] tmp4014;
    wire[7:0] tmp4015;
    wire tmp4016;
    wire[7:0] tmp4017;
    wire[8:0] tmp4018;
    wire[7:0] tmp4019;
    wire[6:0] tmp4020;
    wire[7:0] tmp4021;

    // Combinational
    assign const_758_55 = 55;
    assign const_759_0 = 0;
    assign out4019 = tmp4019;
    assign tmp4010 = {float_a[7]};
    assign tmp4011 = {float_b[7]};
    assign tmp4012 = tmp4010 ^ tmp4011;
    assign tmp4013 = {float_a[6], float_a[5], float_a[4], float_a[3], float_a[2], float_a[1], float_a[0]};
    assign tmp4014 = {float_b[6], float_b[5], float_b[4], float_b[3], float_b[2], float_b[1], float_b[0]};
    assign tmp4015 = tmp4013 + tmp4014;
    assign tmp4016 = {const_759_0};
    assign tmp4017 = {tmp4016, const_758_55};
    assign tmp4018 = tmp4015 - tmp4017;
    assign tmp4019 = tmp4021;
    assign tmp4020 = {tmp4018[6], tmp4018[5], tmp4018[4], tmp4018[3], tmp4018[2], tmp4018[1], tmp4018[0]};
    assign tmp4021 = {tmp4012, tmp4020};

endmodule

