// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out43542);
    input clk;
    input rst;
    input[31:0] float_a;
    input[31:0] float_b;
    output[31:0] out43542;

    wire[30:0] const_6039_1064828928;
    wire const_6040_0;
    wire tmp43533;
    wire tmp43534;
    wire tmp43535;
    wire[30:0] tmp43536;
    wire[30:0] tmp43537;
    wire[31:0] tmp43538;
    wire tmp43539;
    wire[31:0] tmp43540;
    wire[32:0] tmp43541;
    wire[31:0] tmp43542;
    wire[30:0] tmp43543;
    wire[31:0] tmp43544;

    // Combinational
    assign const_6039_1064828928 = 1064828928;
    assign const_6040_0 = 0;
    assign out43542 = tmp43542;
    assign tmp43533 = {float_a[31]};
    assign tmp43534 = {float_b[31]};
    assign tmp43535 = tmp43533 ^ tmp43534;
    assign tmp43536 = {float_a[30], float_a[29], float_a[28], float_a[27], float_a[26], float_a[25], float_a[24], float_a[23], float_a[22], float_a[21], float_a[20], float_a[19], float_a[18], float_a[17], float_a[16], float_a[15], float_a[14], float_a[13], float_a[12], float_a[11], float_a[10], float_a[9], float_a[8], float_a[7], float_a[6], float_a[5], float_a[4], float_a[3], float_a[2], float_a[1], float_a[0]};
    assign tmp43537 = {float_b[30], float_b[29], float_b[28], float_b[27], float_b[26], float_b[25], float_b[24], float_b[23], float_b[22], float_b[21], float_b[20], float_b[19], float_b[18], float_b[17], float_b[16], float_b[15], float_b[14], float_b[13], float_b[12], float_b[11], float_b[10], float_b[9], float_b[8], float_b[7], float_b[6], float_b[5], float_b[4], float_b[3], float_b[2], float_b[1], float_b[0]};
    assign tmp43538 = tmp43536 + tmp43537;
    assign tmp43539 = {const_6040_0};
    assign tmp43540 = {tmp43539, const_6039_1064828928};
    assign tmp43541 = tmp43538 - tmp43540;
    assign tmp43542 = tmp43544;
    assign tmp43543 = {tmp43541[30], tmp43541[29], tmp43541[28], tmp43541[27], tmp43541[26], tmp43541[25], tmp43541[24], tmp43541[23], tmp43541[22], tmp43541[21], tmp43541[20], tmp43541[19], tmp43541[18], tmp43541[17], tmp43541[16], tmp43541[15], tmp43541[14], tmp43541[13], tmp43541[12], tmp43541[11], tmp43541[10], tmp43541[9], tmp43541[8], tmp43541[7], tmp43541[6], tmp43541[5], tmp43541[4], tmp43541[3], tmp43541[2], tmp43541[1], tmp43541[0]};
    assign tmp43544 = {tmp43535, tmp43543};

endmodule

