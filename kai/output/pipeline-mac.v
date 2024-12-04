// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a_in, b_in, clear, write_en, out);
    input clk;
    input rst;
    input[7:0] a_in;
    input[7:0] b_in;
    input clear;
    input write_en;
    output[15:0] out;

    reg[7:0] a_reg = 8'd0;
    reg[15:0] accumulator = 16'd0;
    reg[7:0] b_reg = 8'd0;
    reg[15:0] product_reg = 16'd0;

    wire[15:0] tmp3351;
    wire[16:0] tmp3352;
    wire[15:0] tmp3353;
    wire tmp3354;
    wire[15:0] tmp3355;
    wire[15:0] tmp3356;

    // Combinational
    assign out = accumulator;
    assign tmp3351 = a_reg * b_reg;
    assign tmp3352 = product_reg + accumulator;
    assign tmp3353 = {tmp3352[15], tmp3352[14], tmp3352[13], tmp3352[12], tmp3352[11], tmp3352[10], tmp3352[9], tmp3352[8], tmp3352[7], tmp3352[6], tmp3352[5], tmp3352[4], tmp3352[3], tmp3352[2], tmp3352[1], tmp3352[0]};
    assign tmp3354 = ~write_en;
    assign tmp3355 = write_en ? tmp3353 : accumulator;
    assign tmp3356 = tmp3354 ? accumulator : tmp3355;

    // Registers
    always @(posedge clk or posedge rst)
    begin
        if (rst) begin
            a_reg <= 0;
            accumulator <= 0;
            b_reg <= 0;
            product_reg <= 0;
        end
        else begin
            a_reg <= a_in;
            accumulator <= tmp3356;
            b_reg <= b_in;
            product_reg <= tmp3351;
        end
    end

endmodule

