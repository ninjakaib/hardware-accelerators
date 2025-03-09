// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[7:0] a;
    input[7:0] b;
    output[7:0] result;

    reg[7:0] a_reg = 8'd0;
    reg[7:0] b_reg = 8'd0;

    wire[8:0] tmp16;
    wire[7:0] tmp17;

    // Combinational
    assign result = tmp17;
    assign tmp16 = a_reg + b_reg;
    assign tmp17 = {tmp16[7], tmp16[6], tmp16[5], tmp16[4], tmp16[3], tmp16[2], tmp16[1], tmp16[0]};

    // Registers
    always @(posedge clk)
    begin
        if (rst) begin
            a_reg <= 0;
            b_reg <= 0;
        end
        else begin
            a_reg <= a;
            b_reg <= b;
        end
    end

endmodule

