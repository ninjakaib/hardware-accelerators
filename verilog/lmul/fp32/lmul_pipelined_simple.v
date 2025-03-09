// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out43986);
    input clk;
    input rst;
    input[31:0] float_a;
    input[31:0] float_b;
    output[31:0] out43986;

    reg[31:0] tmp43982;
    reg[31:0] tmp43983;
    reg tmp43984;
    reg[32:0] tmp43985;
    reg[31:0] tmp43986;

    wire[30:0] const_6047_1082654720;
    wire[30:0] const_6048_2147483647;
    wire const_6049_0;
    wire[30:0] const_6050_0;
    wire tmp43987;
    wire tmp43988;
    wire[30:0] tmp43989;
    wire[30:0] tmp43990;
    wire tmp43991;
    wire[31:0] tmp43992;
    wire tmp43993;
    wire[31:0] tmp43994;
    wire[32:0] tmp43995;
    wire[1:0] tmp43996;
    wire[30:0] tmp43997;
    wire tmp43998;
    wire tmp43999;
    wire[30:0] tmp44000;
    wire tmp44001;
    wire[30:0] tmp44002;
    wire[30:0] tmp44003;
    wire[31:0] tmp44004;

    // Combinational
    assign const_6047_1082654720 = 1082654720;
    assign const_6048_2147483647 = 2147483647;
    assign const_6049_0 = 0;
    assign const_6050_0 = 0;
    assign out43986 = tmp43986;
    assign tmp43987 = {tmp43982[31]};
    assign tmp43988 = {tmp43983[31]};
    assign tmp43989 = {tmp43982[30], tmp43982[29], tmp43982[28], tmp43982[27], tmp43982[26], tmp43982[25], tmp43982[24], tmp43982[23], tmp43982[22], tmp43982[21], tmp43982[20], tmp43982[19], tmp43982[18], tmp43982[17], tmp43982[16], tmp43982[15], tmp43982[14], tmp43982[13], tmp43982[12], tmp43982[11], tmp43982[10], tmp43982[9], tmp43982[8], tmp43982[7], tmp43982[6], tmp43982[5], tmp43982[4], tmp43982[3], tmp43982[2], tmp43982[1], tmp43982[0]};
    assign tmp43990 = {tmp43983[30], tmp43983[29], tmp43983[28], tmp43983[27], tmp43983[26], tmp43983[25], tmp43983[24], tmp43983[23], tmp43983[22], tmp43983[21], tmp43983[20], tmp43983[19], tmp43983[18], tmp43983[17], tmp43983[16], tmp43983[15], tmp43983[14], tmp43983[13], tmp43983[12], tmp43983[11], tmp43983[10], tmp43983[9], tmp43983[8], tmp43983[7], tmp43983[6], tmp43983[5], tmp43983[4], tmp43983[3], tmp43983[2], tmp43983[1], tmp43983[0]};
    assign tmp43991 = tmp43987 ^ tmp43988;
    assign tmp43992 = tmp43989 + tmp43990;
    assign tmp43993 = {const_6049_0};
    assign tmp43994 = {tmp43993, const_6047_1082654720};
    assign tmp43995 = tmp43992 + tmp43994;
    assign tmp43996 = {tmp43985[32], tmp43985[31]};
    assign tmp43997 = {tmp43985[30], tmp43985[29], tmp43985[28], tmp43985[27], tmp43985[26], tmp43985[25], tmp43985[24], tmp43985[23], tmp43985[22], tmp43985[21], tmp43985[20], tmp43985[19], tmp43985[18], tmp43985[17], tmp43985[16], tmp43985[15], tmp43985[14], tmp43985[13], tmp43985[12], tmp43985[11], tmp43985[10], tmp43985[9], tmp43985[8], tmp43985[7], tmp43985[6], tmp43985[5], tmp43985[4], tmp43985[3], tmp43985[2], tmp43985[1], tmp43985[0]};
    assign tmp43998 = {tmp43996[1]};
    assign tmp43999 = {tmp43996[0]};
    assign tmp44000 = tmp43999 ? tmp43997 : const_6050_0;
    assign tmp44001 = {tmp43996[0]};
    assign tmp44002 = tmp44001 ? const_6048_2147483647 : const_6048_2147483647;
    assign tmp44003 = tmp43998 ? tmp44002 : tmp44000;
    assign tmp44004 = {tmp43984, tmp44003};

    // Registers
    always @(posedge clk)
    begin
        if (rst) begin
            tmp43982 <= 0;
            tmp43983 <= 0;
            tmp43984 <= 0;
            tmp43985 <= 0;
            tmp43986 <= 0;
        end
        else begin
            tmp43982 <= float_a;
            tmp43983 <= float_b;
            tmp43984 <= tmp43991;
            tmp43985 <= tmp43995;
            tmp43986 <= tmp44004;
        end
    end

endmodule

