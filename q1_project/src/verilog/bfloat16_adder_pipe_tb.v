`include "bfloat16_adder_pipeline.v"

module tb();
    reg clk;
    reg rst;
    reg[15:0] float_a;
    reg[15:0] float_b;
    reg[15:0] gt;
    reg write_enable;
    wire[15:0] result;

    toplevel block(.clk(clk), .rst(rst), .float_a(float_a), .float_b(float_b), .gt(gt), .write_enable(write_enable), .result(result));

    always
        #5 clk = ~clk;

    initial begin
        $dumpfile ("bf16_adder_comb.vcd");
        $dumpvars;

        clk = 0;
        rst = 0;
        block.pipereg_0to1_exp_a = 0;
        block.pipereg_0to1_exp_b = 0;
        block.pipereg_0to1_mant_a = 0;
        block.pipereg_0to1_mant_b = 0;
        block.pipereg_0to1_sign_a = 0;
        block.pipereg_0to1_sign_b = 0;
        block.pipereg_0to1_w_en = 0;
        block.pipereg_1to2_exp_larger = 0;
        block.pipereg_1to2_mant_larger = 0;
        block.pipereg_1to2_mant_smaller = 0;
        block.pipereg_1to2_sign_a = 0;
        block.pipereg_1to2_sign_b = 0;
        block.pipereg_1to2_sign_xor = 0;
        block.pipereg_1to2_signed_shift = 0;
        block.pipereg_1to2_w_en = 0;
        block.pipereg_2to3_aligned_mant_msb = 0;
        block.pipereg_2to3_exp_larger = 0;
        block.pipereg_2to3_guard = 0;
        block.pipereg_2to3_mant_larger = 0;
        block.pipereg_2to3_round = 0;
        block.pipereg_2to3_sign_a = 0;
        block.pipereg_2to3_sign_b = 0;
        block.pipereg_2to3_sign_xor = 0;
        block.pipereg_2to3_signed_shift = 0;
        block.pipereg_2to3_sticky = 0;
        block.pipereg_2to3_w_en = 0;
        block.pipereg_3to4_exp_larger = 0;
        block.pipereg_3to4_guard = 0;
        block.pipereg_3to4_is_neg = 0;
        block.pipereg_3to4_lzc = 0;
        block.pipereg_3to4_mant_sum = 0;
        block.pipereg_3to4_round = 0;
        block.pipereg_3to4_sign_a = 0;
        block.pipereg_3to4_sign_b = 0;
        block.pipereg_3to4_signed_shift = 0;
        block.pipereg_3to4_sticky = 0;
        block.pipereg_3to4_w_en = 0;
        float_a = 16'd16170;
        float_b = 16'd15856;
        gt = 16'd0;
        write_enable = 1'd1;

        #10
        $display("%d", float_out);
        float_a = 16'd16170;
        float_b = 16'd16064;
        gt = 16'd0;
        write_enable = 1'd1;

        #10
        $display("%d", float_out);
        float_a = 16'd16040;
        float_b = 16'd16220;
        gt = 16'd0;
        write_enable = 1'd1;

        #10
        $display("%d", float_out);
        float_a = 16'd15784;
        float_b = 16'd16070;
        gt = 16'd0;
        write_enable = 1'd1;

        #10
        $display("%d", float_out);
        float_a = 16'd16143;
        float_b = 16'd16172;
        gt = 16'd16200;
        write_enable = 1'd1;

        #10
        $display("%d", float_out);
        float_a = 16'd0;
        float_b = 16'd0;
        gt = 16'd16261;
        write_enable = 1'd0;

        #10
        $display("%d", float_out);
        float_a = 16'd0;
        float_b = 16'd0;
        gt = 16'd16280;
        write_enable = 1'd0;

        #10
        $display("%d", float_out);
        float_a = 16'd0;
        float_b = 16'd0;
        gt = 16'd16112;
        write_enable = 1'd0;

        #10
        $display("%d", float_out);
        float_a = 16'd0;
        float_b = 16'd0;
        gt = 16'd16286;
        write_enable = 1'd0;

        #10
        $display("%d", float_out);
        $finish;
    end
endmodule
