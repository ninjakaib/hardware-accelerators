// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out12300);
    input clk;
    input rst;
    input[15:0] float_a;
    input[15:0] float_b;
    output[15:0] out12300;

    reg[15:0] tmp12296;
    reg[15:0] tmp12297;
    reg tmp12298;
    reg[16:0] tmp12299;
    reg[15:0] tmp12300;

    wire[14:0] const_2058_16520;
    wire[14:0] const_2059_32767;
    wire const_2060_0;
    wire[14:0] const_2061_0;
    wire tmp12301;
    wire tmp12302;
    wire[14:0] tmp12303;
    wire[14:0] tmp12304;
    wire tmp12305;
    wire[15:0] tmp12306;
    wire tmp12307;
    wire[15:0] tmp12308;
    wire[16:0] tmp12309;
    wire[1:0] tmp12310;
    wire[14:0] tmp12311;
    wire tmp12312;
    wire tmp12313;
    wire[14:0] tmp12314;
    wire tmp12315;
    wire[14:0] tmp12316;
    wire[14:0] tmp12317;
    wire[15:0] tmp12318;

    // Combinational
    assign const_2058_16520 = 16520;
    assign const_2059_32767 = 32767;
    assign const_2060_0 = 0;
    assign const_2061_0 = 0;
    assign out12300 = tmp12300;
    assign tmp12301 = {tmp12296[15]};
    assign tmp12302 = {tmp12297[15]};
    assign tmp12303 = {tmp12296[14], tmp12296[13], tmp12296[12], tmp12296[11], tmp12296[10], tmp12296[9], tmp12296[8], tmp12296[7], tmp12296[6], tmp12296[5], tmp12296[4], tmp12296[3], tmp12296[2], tmp12296[1], tmp12296[0]};
    assign tmp12304 = {tmp12297[14], tmp12297[13], tmp12297[12], tmp12297[11], tmp12297[10], tmp12297[9], tmp12297[8], tmp12297[7], tmp12297[6], tmp12297[5], tmp12297[4], tmp12297[3], tmp12297[2], tmp12297[1], tmp12297[0]};
    assign tmp12305 = tmp12301 ^ tmp12302;
    assign tmp12306 = tmp12303 + tmp12304;
    assign tmp12307 = {const_2060_0};
    assign tmp12308 = {tmp12307, const_2058_16520};
    assign tmp12309 = tmp12306 + tmp12308;
    assign tmp12310 = {tmp12299[16], tmp12299[15]};
    assign tmp12311 = {tmp12299[14], tmp12299[13], tmp12299[12], tmp12299[11], tmp12299[10], tmp12299[9], tmp12299[8], tmp12299[7], tmp12299[6], tmp12299[5], tmp12299[4], tmp12299[3], tmp12299[2], tmp12299[1], tmp12299[0]};
    assign tmp12312 = {tmp12310[1]};
    assign tmp12313 = {tmp12310[0]};
    assign tmp12314 = tmp12313 ? tmp12311 : const_2061_0;
    assign tmp12315 = {tmp12310[0]};
    assign tmp12316 = tmp12315 ? const_2059_32767 : const_2059_32767;
    assign tmp12317 = tmp12312 ? tmp12316 : tmp12314;
    assign tmp12318 = {tmp12298, tmp12317};

    // Registers
    always @(posedge clk)
    begin
        if (rst) begin
            tmp12296 <= 0;
            tmp12297 <= 0;
            tmp12298 <= 0;
            tmp12299 <= 0;
            tmp12300 <= 0;
        end
        else begin
            tmp12296 <= float_a;
            tmp12297 <= float_b;
            tmp12298 <= tmp12305;
            tmp12299 <= tmp12309;
            tmp12300 <= tmp12318;
        end
    end

endmodule

