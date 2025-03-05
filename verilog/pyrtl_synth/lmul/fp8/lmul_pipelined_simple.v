// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out4109);
    input clk;
    input rst;
    input[7:0] float_a;
    input[7:0] float_b;
    output[7:0] out4109;

    reg[7:0] tmp4105;
    reg[7:0] tmp4106;
    reg tmp4107;
    reg[8:0] tmp4108;
    reg[7:0] tmp4109;

    wire[6:0] const_767_73;
    wire[6:0] const_768_127;
    wire[6:0] const_769_127;
    wire const_770_0;
    wire[6:0] const_771_0;
    wire tmp4110;
    wire tmp4111;
    wire[6:0] tmp4112;
    wire[6:0] tmp4113;
    wire tmp4114;
    wire[7:0] tmp4115;
    wire tmp4116;
    wire[7:0] tmp4117;
    wire[8:0] tmp4118;
    wire[1:0] tmp4119;
    wire[6:0] tmp4120;
    wire tmp4121;
    wire tmp4122;
    wire[6:0] tmp4123;
    wire tmp4124;
    wire[6:0] tmp4125;
    wire[6:0] tmp4126;
    wire[7:0] tmp4127;

    // Combinational
    assign const_767_73 = 73;
    assign const_768_127 = 127;
    assign const_769_127 = 127;
    assign const_770_0 = 0;
    assign const_771_0 = 0;
    assign out4109 = tmp4109;
    assign tmp4110 = {tmp4105[7]};
    assign tmp4111 = {tmp4106[7]};
    assign tmp4112 = {tmp4105[6], tmp4105[5], tmp4105[4], tmp4105[3], tmp4105[2], tmp4105[1], tmp4105[0]};
    assign tmp4113 = {tmp4106[6], tmp4106[5], tmp4106[4], tmp4106[3], tmp4106[2], tmp4106[1], tmp4106[0]};
    assign tmp4114 = tmp4110 ^ tmp4111;
    assign tmp4115 = tmp4112 + tmp4113;
    assign tmp4116 = {const_770_0};
    assign tmp4117 = {tmp4116, const_767_73};
    assign tmp4118 = tmp4115 + tmp4117;
    assign tmp4119 = {tmp4108[8], tmp4108[7]};
    assign tmp4120 = {tmp4108[6], tmp4108[5], tmp4108[4], tmp4108[3], tmp4108[2], tmp4108[1], tmp4108[0]};
    assign tmp4121 = {tmp4119[1]};
    assign tmp4122 = {tmp4119[0]};
    assign tmp4123 = tmp4122 ? tmp4120 : const_771_0;
    assign tmp4124 = {tmp4119[0]};
    assign tmp4125 = tmp4124 ? const_769_127 : const_769_127;
    assign tmp4126 = tmp4121 ? tmp4125 : tmp4123;
    assign tmp4127 = {tmp4107, tmp4126};

    // Registers
    always @(posedge clk)
    begin
        if (rst) begin
            tmp4105 <= 0;
            tmp4106 <= 0;
            tmp4107 <= 0;
            tmp4108 <= 0;
            tmp4109 <= 0;
        end
        else begin
            tmp4105 <= float_a;
            tmp4106 <= float_b;
            tmp4107 <= tmp4114;
            tmp4108 <= tmp4118;
            tmp4109 <= tmp4127;
        end
    end

endmodule

