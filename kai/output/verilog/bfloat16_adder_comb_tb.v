`include "bfloat16_adder_combinatorial.v"

module tb();
    reg clk;
    reg rst;
    reg[15:0] float_a;
    reg[15:0] float_b;
    reg[15:0] gt;
    wire[15:0] float_out;

    toplevel block(.clk(clk), .rst(rst), .float_a(float_a), .float_b(float_b), .gt(gt), .float_out(float_out));

    always
        #5 clk = ~clk;

    initial begin
        $dumpfile ("bf16_adder_comb.vcd");
        $dumpvars;

        clk = 0;
        rst = 0;
        float_a = 16'd16214;
        float_b = 16'd16002;
        gt = 16'd16268;

        #10
        $display("%d", float_out);
        float_a = 16'd16126;
        float_b = 16'd16225;
        gt = 16'd16304;

        #10
        $display("%d", float_out);
        float_a = 16'd16072;
        float_b = 16'd16215;
        gt = 16'd16286;

        #10
        $display("%d", float_out);
        float_a = 16'd16016;
        float_b = 16'd16175;
        gt = 16'd16247;

        #10
        $display("%d", float_out);
        float_a = 16'd15632;
        float_b = 16'd16201;
        gt = 16'd16210;

        #10
        $display("%d", float_out);
        float_a = 16'd15760;
        float_b = 16'd15696;
        gt = 16'd15864;

        #10
        $display("%d", float_out);
        float_a = 16'd16196;
        float_b = 16'd15936;
        gt = 16'd16244;

        #10
        $display("%d", float_out);
        float_a = 16'd16042;
        float_b = 16'd16034;
        gt = 16'd16166;

        #10
        $display("%d", float_out);
        float_a = 16'd16248;
        float_b = 16'd16072;
        gt = 16'd16302;

        #10
        $display("%d", float_out);
        float_a = 16'd16175;
        float_b = 16'd16232;
        gt = 16'd16332;

        #10
        $display("%d", float_out);
        float_a = 16'd15856;
        float_b = 16'd16058;
        gt = 16'd16118;

        #10
        $display("%d", float_out);
        float_a = 16'd16155;
        float_b = 16'd16090;
        gt = 16'd16260;

        #10
        $display("%d", float_out);
        float_a = 16'd16102;
        float_b = 16'd15880;
        gt = 16'd16149;

        #10
        $display("%d", float_out);
        float_a = 16'd16062;
        float_b = 16'd16142;
        gt = 16'd16237;

        #10
        $display("%d", float_out);
        float_a = 16'd16116;
        float_b = 16'd16032;
        gt = 16'd16202;

        #10
        $display("%d", float_out);
        float_a = 16'd16044;
        float_b = 16'd15884;
        gt = 16'd16114;

        #10
        $display("%d", float_out);
        float_a = 16'd15648;
        float_b = 16'd16136;
        gt = 16'd16146;

        #10
        $display("%d", float_out);
        float_a = 16'd16050;
        float_b = 16'd15768;
        gt = 16'd16088;

        #10
        $display("%d", float_out);
        float_a = 16'd16030;
        float_b = 16'd15920;
        gt = 16'd16118;

        #10
        $display("%d", float_out);
        float_a = 16'd16236;
        float_b = 16'd16150;
        gt = 16'd16321;

        #10
        $display("%d", float_out);
        float_a = 16'd15932;
        float_b = 16'd16173;
        gt = 16'd16220;

        #10
        $display("%d", float_out);
        float_a = 16'd16191;
        float_b = 16'd16201;
        gt = 16'd16324;

        #10
        $display("%d", float_out);
        float_a = 16'd15888;
        float_b = 16'd16206;
        gt = 16'd16242;

        #10
        $display("%d", float_out);
        float_a = 16'd15908;
        float_b = 16'd16240;
        gt = 16'd16268;

        #10
        $display("%d", float_out);
        float_a = 16'd16220;
        float_b = 16'd15584;
        gt = 16'd16227;

        #10
        $display("%d", float_out);
        float_a = 16'd16251;
        float_b = 16'd0;
        gt = 16'd16251;

        #10
        $display("%d", float_out);
        float_a = 16'd15980;
        float_b = 16'd16235;
        gt = 16'd16275;

        #10
        $display("%d", float_out);
        float_a = 16'd16070;
        float_b = 16'd16242;
        gt = 16'd16298;

        #10
        $display("%d", float_out);
        float_a = 16'd16161;
        float_b = 16'd16004;
        gt = 16'd16227;

        #10
        $display("%d", float_out);
        float_a = 16'd16150;
        float_b = 16'd16239;
        gt = 16'd16322;

        #10
        $display("%d", float_out);
        float_a = 16'd16201;
        float_b = 16'd16210;
        gt = 16'd16334;

        #10
        $display("%d", float_out);
        float_a = 16'd15824;
        float_b = 16'd16110;
        gt = 16'd16145;

        #10
        $display("%d", float_out);
        float_a = 16'd16227;
        float_b = 16'd15892;
        gt = 16'd16260;

        #10
        $display("%d", float_out);
        float_a = 16'd16192;
        float_b = 16'd15520;
        gt = 16'd16197;

        #10
        $display("%d", float_out);
        float_a = 16'd16096;
        float_b = 16'd15884;
        gt = 16'd16147;

        #10
        $display("%d", float_out);
        float_a = 16'd16166;
        float_b = 16'd15912;
        gt = 16'd16208;

        #10
        $display("%d", float_out);
        float_a = 16'd16227;
        float_b = 16'd16036;
        gt = 16'd16282;

        #10
        $display("%d", float_out);
        float_a = 16'd16092;
        float_b = 16'd15992;
        gt = 16'd16172;

        #10
        $display("%d", float_out);
        float_a = 16'd16226;
        float_b = 16'd16233;
        gt = 16'd16358;

        #10
        $display("%d", float_out);
        float_a = 16'd16235;
        float_b = 16'd16249;
        gt = 16'd16370;

        #10
        $display("%d", float_out);
        float_a = 16'd15584;
        float_b = 16'd16161;
        gt = 16'd16168;

        #10
        $display("%d", float_out);
        float_a = 16'd16227;
        float_b = 16'd16042;
        gt = 16'd16284;

        #10
        $display("%d", float_out);
        float_a = 16'd16205;
        float_b = 16'd16229;
        gt = 16'd16345;

        #10
        $display("%d", float_out);
        float_a = 16'd16210;
        float_b = 16'd15696;
        gt = 16'd16223;

        #10
        $display("%d", float_out);
        float_a = 16'd16249;
        float_b = 16'd15648;
        gt = 16'd16258;

        #10
        $display("%d", float_out);
        float_a = 16'd16207;
        float_b = 16'd15864;
        gt = 16'd16238;

        #10
        $display("%d", float_out);
        float_a = 16'd16068;
        float_b = 16'd16152;
        gt = 16'd16250;

        #10
        $display("%d", float_out);
        float_a = 16'd16251;
        float_b = 16'd15744;
        gt = 16'd16262;

        #10
        $display("%d", float_out);
        float_a = 16'd16046;
        float_b = 16'd16198;
        gt = 16'd16270;

        #10
        $display("%d", float_out);
        float_a = 16'd16080;
        float_b = 16'd16253;
        gt = 16'd16306;

        #10
        $display("%d", float_out);
        float_a = 16'd16192;
        float_b = 16'd16190;
        gt = 16'd16319;

        #10
        $display("%d", float_out);
        float_a = 16'd16203;
        float_b = 16'd16106;
        gt = 16'd16288;

        #10
        $display("%d", float_out);
        float_a = 16'd16020;
        float_b = 16'd16190;
        gt = 16'd16260;

        #10
        $display("%d", float_out);
        float_a = 16'd15968;
        float_b = 16'd16010;
        gt = 16'd16122;

        #10
        $display("%d", float_out);
        float_a = 16'd16020;
        float_b = 16'd16161;
        gt = 16'd16235;

        #10
        $display("%d", float_out);
        float_a = 16'd15904;
        float_b = 16'd16147;
        gt = 16'd16187;

        #10
        $display("%d", float_out);
        float_a = 16'd15800;
        float_b = 16'd16092;
        gt = 16'd16133;

        #10
        $display("%d", float_out);
        float_a = 16'd16042;
        float_b = 16'd16161;
        gt = 16'd16246;

        #10
        $display("%d", float_out);
        float_a = 16'd16052;
        float_b = 16'd16190;
        gt = 16'd16268;

        #10
        $display("%d", float_out);
        float_a = 16'd16168;
        float_b = 16'd16158;
        gt = 16'd16291;

        #10
        $display("%d", float_out);
        float_a = 16'd16214;
        float_b = 16'd16136;
        gt = 16'd16303;

        #10
        $display("%d", float_out);
        float_a = 16'd16180;
        float_b = 16'd15892;
        gt = 16'd16217;

        #10
        $display("%d", float_out);
        float_a = 16'd16159;
        float_b = 16'd16199;
        gt = 16'd16307;

        #10
        $display("%d", float_out);
        float_a = 16'd16060;
        float_b = 16'd16131;
        gt = 16'd16225;

        #10
        $display("%d", float_out);
        float_a = 16'd16242;
        float_b = 16'd16208;
        gt = 16'd16353;

        #10
        $display("%d", float_out);
        float_a = 16'd16249;
        float_b = 16'd16176;
        gt = 16'd16340;

        #10
        $display("%d", float_out);
        float_a = 16'd16162;
        float_b = 16'd15912;
        gt = 16'd16204;

        #10
        $display("%d", float_out);
        float_a = 16'd16072;
        float_b = 16'd16175;
        gt = 16'd16266;

        #10
        $display("%d", float_out);
        float_a = 16'd16217;
        float_b = 16'd15944;
        gt = 16'd16262;

        #10
        $display("%d", float_out);
        float_a = 16'd16148;
        float_b = 16'd16226;
        gt = 16'd16315;

        #10
        $display("%d", float_out);
        float_a = 16'd16188;
        float_b = 16'd16157;
        gt = 16'd16300;

        #10
        $display("%d", float_out);
        float_a = 16'd16238;
        float_b = 16'd16118;
        gt = 16'd16308;

        #10
        $display("%d", float_out);
        float_a = 16'd16183;
        float_b = 16'd15856;
        gt = 16'd16213;

        #10
        $display("%d", float_out);
        float_a = 16'd16142;
        float_b = 16'd16220;
        gt = 16'd16309;

        #10
        $display("%d", float_out);
        float_a = 16'd16170;
        float_b = 16'd16110;
        gt = 16'd16272;

        #10
        $display("%d", float_out);
        float_a = 16'd16036;
        float_b = 16'd15996;
        gt = 16'd16145;

        #10
        $display("%d", float_out);
        float_a = 16'd16218;
        float_b = 16'd16074;
        gt = 16'd16288;

        #10
        $display("%d", float_out);
        float_a = 16'd16226;
        float_b = 16'd16166;
        gt = 16'd16324;

        #10
        $display("%d", float_out);
        float_a = 16'd16255;
        float_b = 16'd15488;
        gt = 16'd16258;

        #10
        $display("%d", float_out);
        float_a = 16'd16018;
        float_b = 16'd15892;
        gt = 16'd16092;

        #10
        $display("%d", float_out);
        float_a = 16'd16185;
        float_b = 16'd16094;
        gt = 16'd16276;

        #10
        $display("%d", float_out);
        float_a = 16'd16058;
        float_b = 16'd15988;
        gt = 16'd16154;

        #10
        $display("%d", float_out);
        float_a = 16'd16074;
        float_b = 16'd16179;
        gt = 16'd16268;

        #10
        $display("%d", float_out);
        float_a = 16'd16131;
        float_b = 16'd16220;
        gt = 16'd16304;

        #10
        $display("%d", float_out);
        float_a = 16'd15520;
        float_b = 16'd16070;
        gt = 16'd16080;

        #10
        $display("%d", float_out);
        float_a = 16'd16227;
        float_b = 16'd15696;
        gt = 16'd16240;

        #10
        $display("%d", float_out);
        float_a = 16'd16137;
        float_b = 16'd16240;
        gt = 16'd16316;

        #10
        $display("%d", float_out);
        float_a = 16'd16227;
        float_b = 16'd16138;
        gt = 16'd16310;

        #10
        $display("%d", float_out);
        float_a = 16'd16062;
        float_b = 16'd16145;
        gt = 16'd16240;

        #10
        $display("%d", float_out);
        float_a = 16'd16177;
        float_b = 16'd16116;
        gt = 16'd16278;

        #10
        $display("%d", float_out);
        float_a = 16'd16182;
        float_b = 16'd16199;
        gt = 16'd16318;

        #10
        $display("%d", float_out);
        float_a = 16'd16206;
        float_b = 16'd16233;
        gt = 16'd16348;

        #10
        $display("%d", float_out);
        float_a = 16'd16066;
        float_b = 16'd15972;
        gt = 16'd16154;

        #10
        $display("%d", float_out);
        float_a = 16'd16220;
        float_b = 16'd15864;
        gt = 16'd16251;

        #10
        $display("%d", float_out);
        float_a = 16'd16122;
        float_b = 16'd16236;
        gt = 16'd16308;

        #10
        $display("%d", float_out);
        float_a = 16'd16014;
        float_b = 16'd15876;
        gt = 16'd16080;

        #10
        $display("%d", float_out);
        float_a = 16'd15616;
        float_b = 16'd15956;
        gt = 16'd15988;

        #10
        $display("%d", float_out);
        float_a = 16'd16010;
        float_b = 16'd15988;
        gt = 16'd16130;

        #10
        $display("%d", float_out);
        float_a = 16'd16225;
        float_b = 16'd16080;
        gt = 16'd16292;

        #10
        $display("%d", float_out);
        float_a = 16'd16169;
        float_b = 16'd16050;
        gt = 16'd16257;

        #10
        $display("%d", float_out);
        $finish;
    end
endmodule
