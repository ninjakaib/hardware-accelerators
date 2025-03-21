// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out1282, out1284, out1285, out1286, out1287);
    input clk;
    input rst;
    input[7:0] float_a;
    input[7:0] float_b;
    output out1282;
    output[4:0] out1284;
    output[3:0] out1285;
    output[3:0] out1286;
    output[3:0] out1287;

    wire const_238_1;
    wire const_239_1;
    wire const_240_1;
    wire const_241_0;
    wire const_242_0;
    wire tmp1268;
    wire tmp1269;
    wire tmp1270;
    wire tmp1271;
    wire[3:0] tmp1272;
    wire[3:0] tmp1273;
    wire[3:0] tmp1274;
    wire[3:0] tmp1275;
    wire[3:0] tmp1276;
    wire[3:0] tmp1277;
    wire[2:0] tmp1278;
    wire[3:0] tmp1279;
    wire[2:0] tmp1280;
    wire[3:0] tmp1281;
    wire tmp1282;
    wire[4:0] tmp1283;
    wire[4:0] tmp1284;
    wire[3:0] tmp1285;
    wire[3:0] tmp1286;
    wire[3:0] tmp1287;
    wire tmp1288;
    wire[4:0] tmp1289;
    wire tmp1290;
    wire tmp1291;
    wire[3:0] tmp1292;
    wire[3:0] tmp1293;
    wire tmp1294;
    wire[3:0] tmp1295;
    wire[3:0] tmp1296;
    wire[2:0] tmp1297;
    wire[3:0] tmp1298;
    wire[4:0] tmp1299;
    wire[3:0] tmp1300;
    wire[4:0] tmp1301;
    wire tmp1302;
    wire[4:0] tmp1303;
    wire[4:0] tmp1304;
    wire tmp1305;
    wire[3:0] tmp1306;
    wire tmp1307;
    wire[3:0] tmp1308;

    // Combinational
    assign const_238_1 = 1;
    assign const_239_1 = 1;
    assign const_240_1 = 1;
    assign const_241_0 = 0;
    assign const_242_0 = 0;
    assign out1282 = tmp1282;
    assign out1284 = tmp1284;
    assign out1285 = tmp1285;
    assign out1286 = tmp1286;
    assign out1287 = tmp1287;
    assign tmp1268 = tmp1270;
    assign tmp1269 = tmp1271;
    assign tmp1270 = {float_a[7]};
    assign tmp1271 = {float_b[7]};
    assign tmp1272 = tmp1274;
    assign tmp1273 = tmp1275;
    assign tmp1274 = {float_a[6], float_a[5], float_a[4], float_a[3]};
    assign tmp1275 = {float_b[6], float_b[5], float_b[4], float_b[3]};
    assign tmp1276 = tmp1279;
    assign tmp1277 = tmp1281;
    assign tmp1278 = {float_a[2], float_a[1], float_a[0]};
    assign tmp1279 = {const_238_1, tmp1278};
    assign tmp1280 = {float_b[2], float_b[1], float_b[0]};
    assign tmp1281 = {const_239_1, tmp1280};
    assign tmp1282 = tmp1288;
    assign tmp1283 = tmp1289;
    assign tmp1284 = tmp1304;
    assign tmp1285 = tmp1292;
    assign tmp1286 = tmp1306;
    assign tmp1287 = tmp1308;
    assign tmp1288 = tmp1268 ^ tmp1269;
    assign tmp1289 = tmp1272 - tmp1273;
    assign tmp1290 = {tmp1283[4]};
    assign tmp1291 = {tmp1283[4]};
    assign tmp1292 = tmp1291 ? tmp1273 : tmp1272;
    assign tmp1293 = {tmp1283[3], tmp1283[2], tmp1283[1], tmp1283[0]};
    assign tmp1294 = {tmp1283[4]};
    assign tmp1295 = {tmp1283[3], tmp1283[2], tmp1283[1], tmp1283[0]};
    assign tmp1296 = ~tmp1295;
    assign tmp1297 = {const_241_0, const_241_0, const_241_0};
    assign tmp1298 = {tmp1297, const_240_1};
    assign tmp1299 = tmp1296 + tmp1298;
    assign tmp1300 = {tmp1299[3], tmp1299[2], tmp1299[1], tmp1299[0]};
    assign tmp1301 = {tmp1294, tmp1300};
    assign tmp1302 = {const_242_0};
    assign tmp1303 = {tmp1302, tmp1293};
    assign tmp1304 = tmp1290 ? tmp1301 : tmp1303;
    assign tmp1305 = {tmp1283[4]};
    assign tmp1306 = tmp1305 ? tmp1276 : tmp1277;
    assign tmp1307 = {tmp1283[4]};
    assign tmp1308 = tmp1307 ? tmp1277 : tmp1276;

endmodule

